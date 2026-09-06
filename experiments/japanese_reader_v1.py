#!/usr/bin/env python3
"""Developmental Japanese reading loop -- runs alongside the English worker.

One cycle:
  1. select the next book from the reading curriculum (its ZPD), or fetch more
     Japanese children's stories when the shelf runs thin
  2. extract its events (japanese_event_v1), score comprehension against a
     model trained on the books read so far (reading_comprehension_v1)
  3. record the reading: grow vocabulary, graduate / reread / shelve
  4. maybe advance the reading level and un-shelve what is now in reach
  5. run the frozen-benchmark comprehension measurement for the capability claim

State lives in .local/reading-*.json and is completely separate from the
English pipeline's .local files, so the two never interfere.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from web_cache import WEB_CACHE
import japanese_corpus_v1 as corpus
import japanese_event_v1 as jevent
import reading_curriculum_v1 as curriculum
import reading_comprehension_v1 as comprehension

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME = ROOT / ".local"
CURRICULUM_FILE = "reading-curriculum.json"
EVENTS_FILE = "reading-events.json"
COMPREHENSION_FILE = "reading-comprehension.json"
STATUS_FILE = "reading-status.json"
STOP_FILE = "READING_STOP"
SHELF_LOW_WATER = 3           # in-rotation books below this -> fetch more
FETCH_BUDGET = 12
FETCH_COOLDOWN = 5           # cycles to wait between shelf-widening fetches


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def _events_of(text: str) -> list[dict]:
    return [e.__dict__ for e in jevent.extract_story(text)]


def _fetch_more_books(cur: dict, cycle: int) -> int:
    """Widen the shelf: the kernel first, then Aozora children's authors."""
    WEB_CACHE.set_network_budget(FETCH_BUDGET)
    have = {b["url"] for b in cur["shelf"].values()}
    fetched: list[dict] = []
    for title in corpus.KERNEL:
        if len(fetched) >= 6:
            break
        story = corpus.fetch(title)
        if story and story.url not in have:
            fetched.append(story)
    if len(fetched) < 3:
        for name, pid in corpus.AOZORA_AUTHORS.items():
            for wtitle, wurl in corpus.aozora_author_works(pid, limit=8):
                if wurl in have or any(f.url == wurl for f in fetched):
                    continue
                story = corpus.fetch_aozora(wurl)
                if story:
                    fetched.append(story)
                if len(fetched) >= 8:
                    break
            if len(fetched) >= 8:
                break
    books = [{"title": s.title, "url": s.url, "source": "ja",
              "text": s.text, "event_count": len(_events_of(s.text))} for s in fetched]
    added = curriculum.register_books(cur, books, cycle)
    # keep the raw events for the books we just added
    if added:
        events = _read_events()
        for s in fetched:
            bid = curriculum._book_id(s.url, s.title)
            events.setdefault(bid, _events_of(s.text))
        _write_events(events)
    return added


_events_path: Path | None = None


def _read_events() -> dict:
    return _read(_events_path)


def _write_events(value: dict) -> None:
    _write(_events_path, value)


def run_once(runtime: Path) -> dict:
    global _events_path
    _events_path = runtime / EVENTS_FILE
    cur = _read(runtime / CURRICULUM_FILE) or curriculum.empty_curriculum()
    prev_comp = _read(runtime / COMPREHENSION_FILE)
    cycle = cur.get("cycle", 0) + 1
    cur["cycle"] = cycle
    events_store = _read_events()

    in_rotation = sum(1 for b in cur["shelf"].values() if b["status"] == "in_rotation")
    fetched = 0
    cooled = cycle - cur.get("last_fetch_cycle", -FETCH_COOLDOWN) >= FETCH_COOLDOWN
    if in_rotation < SHELF_LOW_WATER and cooled:
        fetched = _fetch_more_books(cur, cycle)
        cur["last_fetch_cycle"] = cycle
        events_store = _read_events()

    # a comprehension model from every story read so far
    read_events = [ev for bid, ev in events_store.items()
                   if cur["shelf"].get(bid, {}).get("times_read", 0) > 0 and len(ev) >= 3]
    model = comprehension.ComprehensionModel().fit(read_events) if len(read_events) >= 5 else None

    book_id = curriculum.retention_check_due(cur, cycle) or curriculum.select_next_book(cur)
    reading = {"status": "no_book"}
    if book_id:
        book = cur["shelf"][book_id]
        events = events_store.get(book_id) or _events_of(book.get("text", ""))
        events_store.setdefault(book_id, events)
        reading = curriculum.record_reading(cur, book_id, events, cycle, model=model)
        reading["title"] = book["title"]
        reading["level"] = book["estimated_level"]

    advance = curriculum.maybe_advance_level(cur, cycle)

    # frozen-benchmark capability measurement over all stories with events
    all_stories = [{"url": cur["shelf"][bid]["url"], "events": ev}
                   for bid, ev in events_store.items() if bid in cur["shelf"] and len(ev) >= 3]
    comp_report = comprehension.evaluate_comprehension(all_stories, prev_comp)

    _write(runtime / CURRICULUM_FILE, cur)
    _write_events(events_store)
    _write(runtime / COMPREHENSION_FILE, comp_report)
    status = {
        "heartbeat": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cycle": cycle, "books_fetched": fetched,
        "reading": reading, "level_advance": advance,
        "curriculum": curriculum.summary(cur),
        "comprehension": {k: comp_report.get(k) for k in
                          ("status", "comprehension_score", "consequence",
                           "consequence_baseline", "consequence_z", "beats_baseline",
                           "comprehension_trend", "test_stories")},
    }
    _write(runtime / STATUS_FILE, status)
    return status


def supervise(runtime: Path, interval: float) -> None:
    stop = runtime / STOP_FILE
    stop.unlink(missing_ok=True)
    while not stop.exists():
        try:
            status = run_once(runtime)
        except Exception as error:
            status = {"heartbeat": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "error": f"{type(error).__name__}: {error}",
                      "traceback": traceback.format_exc()[-2000:]}
            _write(runtime / STATUS_FILE, status)
        if stop.exists():
            break
        time.sleep(max(1.0, interval))


def render_status(runtime: Path) -> str:
    s = _read(runtime / STATUS_FILE)
    c = s.get("curriculum", {})
    comp = s.get("comprehension", {})
    lines = ["Noise 日本語読書", "=" * 34,
             f"最終更新   : {s.get('heartbeat', '不明')}",
             f"サイクル   : {s.get('cycle', 0)}",
             f"読解レベル : {c.get('level')}（{c.get('milestone')} / {c.get('reading_age')}）",
             f"本棚       : 回転中 {c.get('in_rotation')}、卒業 {c.get('graduated')}、"
             f"棚上げ {c.get('shelved_above_level', 0)}＋手詰まり {c.get('shelved_stuck', 0)}",
             f"語彙       : seen {c.get('vocabulary_seen')}／used {c.get('vocabulary_used')}／"
             f"explained {c.get('vocabulary_explained')}（explained目標 {c.get('explained_ratio_target')}）",
             f"直近の読書 : {s.get('reading', {}).get('title', '-')} "
             f"→ {s.get('reading', {}).get('status', '-')} "
             f"comp={s.get('reading', {}).get('comprehension')}",
             f"理解(固定) : score={comp.get('comprehension_score')} "
             f"帰結={comp.get('consequence')}/基準{comp.get('consequence_baseline')} "
             f"z={comp.get('consequence_z')} 基準超え={comp.get('beats_baseline')} "
             f"傾向={comp.get('comprehension_trend')}"]
    if s.get("level_advance", {}).get("advanced"):
        lines.append(f"★ レベル上昇 → {s['level_advance']['level']}")
    if s.get("error"):
        lines.append(f"エラー: {s['error']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "supervise", "status", "stop"))
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    args.runtime.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        print(json.dumps(run_once(args.runtime), ensure_ascii=False, indent=1))
    elif args.command == "supervise":
        supervise(args.runtime, args.interval)
    elif args.command == "status":
        print(render_status(args.runtime))
    elif args.command == "stop":
        (args.runtime / STOP_FILE).touch()
        print("reading stop requested")


if __name__ == "__main__":
    main()
