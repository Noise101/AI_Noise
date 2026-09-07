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
import japanese_retell_v1 as retell
import japanese_sequence_v1 as sequence
import caregiver_v1 as caregiver
import reading_llm_v1 as reading_llm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME = ROOT / ".local"
CURRICULUM_FILE = "reading-curriculum.json"
EVENTS_FILE = "reading-events.json"
COMPREHENSION_FILE = "reading-comprehension.json"
RETELL_FILE = "reading-retelling.json"
SEQUENCE_FILE = "reading-sequence.json"
CAREGIVER_FILE = "caregiver.json"
STATUS_FILE = "reading-status.json"
SEQUENCE_TRAIN_SECONDS = 5.0
STOP_FILE = "READING_STOP"
SHELF_LOW_WATER = 4           # in-rotation books below this -> fetch more
FETCH_BUDGET = 20           # network requests per shelf-widening pass
FETCH_TARGET = 10           # books to add per pass
FETCH_COOLDOWN = 4           # cycles to wait between shelf-widening fetches
AOZORA_LEVEL_MARGIN = 3.0   # skip Aozora works this far above the reading level
AOZORA_WORKS_PER_AUTHOR = 10


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


def _scaffold_totals(scaffolded: dict) -> dict:
    return {"attempted": len(scaffolded),
            "verified": sum(1 for v in scaffolded.values() if v.get("status") == "simplified")}


def _within_reach(text: str, level: float) -> bool:
    """Keep Aozora acquisition near the reading level so the shelf does not
    fill with 家なき子 / 明日 while the reader is on picture books."""
    try:
        est = curriculum.text_difficulty(text, 0, set())["estimated_level"]
    except Exception:
        return True
    return est <= level + AOZORA_LEVEL_MARGIN


def _fetch_more_books(cur: dict, cycle: int) -> int:
    """Widen the shelf: verified Aesop/folktale kernel first, then Aozora
    children's authors -- level-gated, paced, small batches."""
    WEB_CACHE.set_network_budget(FETCH_BUDGET)
    have = {b["url"] for b in cur["shelf"].values()}
    level = cur.get("level", 1.5)
    fetched: list = []

    def budget_left() -> bool:
        return WEB_CACHE.remaining_network_budget() != 0

    for title in corpus.kernel_titles():
        if len(fetched) >= FETCH_TARGET or not budget_left():
            break
        story = corpus.fetch(title)
        if story and story.url not in have:
            fetched.append(story)

    if len(fetched) < FETCH_TARGET and budget_left():
        # rotate which author we start from so every author gets sampled over
        # successive passes, not just the first few each time
        authors = list(corpus.AOZORA_AUTHORS.items())
        start = cur.get("_aozora_cursor", 0) % max(1, len(authors))
        cur["_aozora_cursor"] = start + 1
        for _name, pid in authors[start:] + authors[:start]:
            for _wtitle, wurl in corpus.aozora_author_works(pid, limit=AOZORA_WORKS_PER_AUTHOR):
                if len(fetched) >= FETCH_TARGET or not budget_left():
                    break
                if wurl in have or any(f.url == wurl for f in fetched):
                    continue
                story = corpus.fetch_aozora(wurl)
                if story and _within_reach(story.text, level):
                    fetched.append(story)
            if len(fetched) >= FETCH_TARGET or not budget_left():
                break
    books = []
    fetched_events = {}
    for s in fetched:
        evs = _events_of(s.text)
        fetched_events[s.url] = evs
        books.append({"title": s.title, "url": s.url, "source": "ja", "text": s.text,
                      "event_count": len(evs),
                      "verbs": [e.get("verb", "") for e in evs]})
    added = curriculum.register_books(cur, books, cycle)
    # keep the raw events for the books we just added
    if added:
        events = _read_events()
        for s in fetched:
            bid = curriculum._book_id(s.url, s.title)
            events.setdefault(bid, fetched_events[s.url])
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
    prev_retell = _read(runtime / RETELL_FILE)
    prev_seq = _read(runtime / SEQUENCE_FILE)
    cycle = cur.get("cycle", 0) + 1
    cur["cycle"] = cycle
    events_store = _read_events()

    # a better parser deserves a fresh reading of the whole shelf: drop the event
    # cache, put books it had set aside back in rotation, and re-walk from the
    # easiest level.
    from japanese_event_v1 import PARSER_VERSION
    CORPUS_VERSION = 2                            # bump when _modernise changes
    if cur.get("corpus_version", 0) < CORPUS_VERSION:
        for b in cur["shelf"].values():
            if b.get("text"):
                b["text"] = corpus._modernise(b["text"])
        cur["corpus_version"] = CORPUS_VERSION
        events_store = {}
    if cur.get("events_parser_version", 0) < PARSER_VERSION:
        events_store = {}
        cur["events_parser_version"] = PARSER_VERSION
    for bid in curriculum.reevaluate_stale_parses(cur):
        events_store.pop(bid, None)
    curriculum.reset_level_for_new_parser(cur, cycle)

    in_rotation = sum(1 for b in cur["shelf"].values() if b["status"] == "in_rotation")
    reachable = in_rotation + sum(1 for b in cur["shelf"].values()
                                 if b["status"] in ("shelved_above_level", "shelved_stuck"))
    fetched = 0
    cooled = cycle - cur.get("last_fetch_cycle", -FETCH_COOLDOWN) >= FETCH_COOLDOWN
    # honour the cooldown normally, but never sit with an empty shelf
    if (in_rotation < SHELF_LOW_WATER and cooled) or reachable == 0:
        fetched = _fetch_more_books(cur, cycle)
        cur["last_fetch_cycle"] = cycle
        events_store = _read_events()

    book_id = curriculum.retention_check_due(cur, cycle) or curriculum.select_next_book(cur)
    reading = {"status": "no_book"}
    scaffold = None
    model = None                                  # caregiver batch reads this even with no book
    if book_id:
        # a per-book comprehension model must not train on the book it scores
        others = [ev for bid, ev in events_store.items()
                  if bid != book_id and cur["shelf"].get(bid, {}).get("times_read", 0) > 0
                  and len(ev) >= 3]
        model = comprehension.ComprehensionModel().fit(others) if len(others) >= 5 else None
        book = cur["shelf"][book_id]
        events = events_store.get(book_id) or _events_of(book.get("text", ""))
        events_store.setdefault(book_id, events)
        # local-LLM reading scaffold: a story Noise cannot parse (a caregiver
        # would paraphrase the hard passages) -- verified to parse and preserve
        # content, then used only for this book's comprehension, never the
        # frozen benchmark or the RNN.
        tried = cur.setdefault("llm_scaffolded", {})
        parseable = sum(1 for e in events if e.get("verb"))
        if (parseable < reading_llm.MIN_EVENTS and book_id not in tried
                and book.get("text") and reading_llm.OllamaReader().available()):
            base_known = {w for w, info in cur.get("known_words", {}).items()
                          if info.get("books", 0) >= 1}
            scaffold = reading_llm.simplify_story(book["text"], base_known_words=base_known)
            tried[book_id] = {"status": scaffold["status"],
                              "event_count": scaffold.get("event_count", 0),
                              "content_kept": scaffold.get("content_kept"), "cycle": cycle}
            if scaffold.get("verified"):
                events = scaffold["events"]
                events_store[book_id] = events
                book["scaffolded"] = True
        # per-book scoring may use the disclosed morphological teacher (evidence
        # score 0); events_store keeps the heuristic parse for the frozen
        # benchmarks and the RNN, exactly like the LLM scaffold split
        scored = events if book.get("scaffolded") else jevent.refine_event_dicts(events)
        reading = curriculum.record_reading(cur, book_id, scored, cycle, model=model)
        reading["title"] = book["title"]
        reading["level"] = book["estimated_level"]
        # what Noise understood, in its own (template) words
        retold = retell.retell(scored, max_sentences=10)
        reading["retelling"] = retold
        reading["retelling_score"] = retell.score_retelling(scored, retold)

    advance = curriculum.maybe_advance_level(cur, cycle)

    # frozen-benchmark capability measurements -- real reading only, never the
    # LLM-scaffolded books
    real = {bid for bid, b in cur["shelf"].items() if not b.get("scaffolded")}
    all_stories = [{"url": cur["shelf"][bid]["url"], "events": ev}
                   for bid, ev in events_store.items()
                   if bid in real and len(ev) >= 3]
    comp_report = comprehension.evaluate_comprehension(all_stories, prev_comp)
    retell_report = retell.evaluate_retelling(all_stories, prev_retell)

    # character RNN over the sentences read so far (continuous capability signal)
    seq_texts: dict[str, str] = {}
    for bid, ev in events_store.items():
        if bid in real and ev:
            url = cur["shelf"][bid]["url"]
            seq_texts[url] = seq_texts.get(url, "") + "".join(e.get("sentence", "") for e in ev)
    seq_report = sequence.train_and_evaluate(seq_texts, prev_seq, SEQUENCE_TRAIN_SECONDS)

    # a free-generation retelling of the book just read, primed on its opening
    if book_id and seq_report.get("state"):
        opening = (events[0].get("sentence") if events else "") or "むかしむかし"
        free = retell.free_retell(seq_report["state"], opening[:12])
        if free:
            reading["free_retelling"] = free
            reading["free_retelling_score"] = retell.score_retelling(events, free)

    # low-effort caregiver check: a small batch of multiple-choice / yes-no
    # questions about recently-read books, every CAREGIVER_INTERVAL cycles
    care_state = _read(runtime / CAREGIVER_FILE) or caregiver.empty_state()
    caregiver.expire_if_stale(care_state, cycle)
    if caregiver.due(care_state, cycle):
        current_fidelity = (reading.get("retelling_score") or {}).get("fidelity")
        recent = [{"title": b["title"], "url": b["url"],
                   # the caregiver judges the same (teacher-refined) retelling
                   # that record_reading scored
                   "events": jevent.refine_event_dicts(events_store.get(bid, [])),
                   "fidelity": current_fidelity if bid == book_id else None,
                   "_last": b.get("last_read_cycle") or 0}
                  for bid, b in cur["shelf"].items()
                  if b.get("times_read", 0) > 0 and len(events_store.get(bid, [])) >= 3]
        recent.sort(key=lambda r: -r["_last"])
        questions = caregiver.generate_questions(recent, cycle, model)
        if questions:
            caregiver.open_batch(care_state, questions, cycle)
    _write(runtime / CAREGIVER_FILE, care_state)

    _write(runtime / CURRICULUM_FILE, cur)
    _write_events(events_store)
    _write(runtime / COMPREHENSION_FILE, comp_report)
    _write(runtime / RETELL_FILE, retell_report)
    _write(runtime / SEQUENCE_FILE, seq_report)
    status = {
        "heartbeat": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cycle": cycle, "books_fetched": fetched,
        "reading": reading, "level_advance": advance,
        "curriculum": curriculum.summary(cur),
        "comprehension": {k: comp_report.get(k) for k in
                          ("status", "comprehension_score", "consequence",
                           "consequence_baseline", "consequence_z", "beats_baseline",
                           "comprehension_trend", "test_stories")},
        "retelling": {k: retell_report.get(k) for k in
                      ("status", "fidelity", "fidelity_baseline", "gain_z",
                       "beats_baseline", "retelling_trend", "test_stories")},
        "sequence": {k: seq_report.get(k) for k in
                     ("status", "held_out_bits_per_char", "baseline_bits_per_char",
                      "improvement_bits", "improvement_z", "beats_char_baseline",
                      "perplexity_trend", "steps_trained")},
        "sequence_sample": (seq_report.get("samples") or [""])[0],
        "caregiver": caregiver.summary(care_state),
        "caregiver_questions": [q["prompt"] for q in care_state.get("pending", [])],
        "llm_scaffold": ({"book": reading.get("title"), **{k: scaffold.get(k) for k in
                          ("status", "event_count", "content_kept", "length_ratio")}}
                         if scaffold else None),
        "llm_scaffold_totals": _scaffold_totals(cur.get("llm_scaffolded", {})),
    }
    _write(runtime / STATUS_FILE, status)
    return status


def answer_questions(runtime: Path, raw: str) -> dict:
    path = runtime / CAREGIVER_FILE
    state = _read(path) or caregiver.empty_state()
    if not state.get("pending"):
        return {"status": "no_pending_questions"}
    result = caregiver.apply_answers(state, raw)
    _write(path, state)
    result["status"] = "recorded"
    result.update(caregiver.summary(state))
    return result


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
    ret = s.get("retelling", {})
    seq = s.get("sequence", {})
    reading = s.get("reading", {})
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
             f"傾向={comp.get('comprehension_trend')}",
             f"再話(固定) : 忠実度={ret.get('fidelity')}/基準{ret.get('fidelity_baseline')} "
             f"z={ret.get('gain_z')} 基準超え={ret.get('beats_baseline')} "
             f"傾向={ret.get('retelling_trend')}",
             f"文字RNN   : {seq.get('held_out_bits_per_char')}bpc/基準{seq.get('baseline_bits_per_char')} "
             f"z={seq.get('improvement_z')} 基準超え={seq.get('beats_char_baseline')} "
             f"傾向={seq.get('perplexity_trend')} 学習{seq.get('steps_trained')}歩"]
    if s.get("sequence_sample"):
        lines.append(f"  生成: {s['sequence_sample'][:60]}")
    retold = reading.get("retelling")
    if retold:
        rs = reading.get("retelling_score", {})
        lines.append(f"再話(定型)  : （再現{rs.get('recall')}／順序{rs.get('order_correlation')}）")
        lines.append(f"  {retold[:120]}")
    free = reading.get("free_retelling")
    if free:
        fs = reading.get("free_retelling_score", {})
        lines.append(f"再話(RNN生成): （再現{fs.get('recall')}／忠実度{fs.get('fidelity')}）")
        lines.append(f"  {free[:120]}")
    if s.get("level_advance", {}).get("advanced"):
        lines.append(f"★ レベル上昇 → {s['level_advance']['level']}")
    sc = s.get("llm_scaffold_totals", {})
    if sc.get("attempted"):
        lines.append(f"LLM読解補助: {sc.get('verified')}/{sc.get('attempted')}冊 "
                     f"(解析できない物語をローカルLLMが平易化、検証済みのみ採用・固定検証には不使用)")
    care = s.get("caregiver", {})
    if care.get("human_checked"):
        lines.append(f"保護者確認 : {care['human_checked']}問回答済み"
                     f"（物語一致 {care.get('human_story_agreement')}／モデル一致 {care.get('human_model_agreement')}）")
    pending = s.get("caregiver_questions") or []
    if pending:
        lines.append(f"── 質問 {len(pending)}件（答えるには: japanese_reader_v1.py answer \"…\"）──")
        for i, q in enumerate(pending, 1):
            lines.append(f"  {i}. {q}")
    if s.get("error"):
        lines.append(f"エラー: {s['error']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "supervise", "status", "stop", "answer"))
    parser.add_argument("reply", nargs="?", default="",
                        help="for 'answer': positional answers, e.g. \"2 はい 1\"")
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
    elif args.command == "answer":
        print(json.dumps(answer_questions(args.runtime, args.reply), ensure_ascii=False, indent=1))
    elif args.command == "stop":
        (args.runtime / STOP_FILE).touch()
        print("reading stop requested")


if __name__ == "__main__":
    main()
