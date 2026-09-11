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
import japanese_benchmark_v1 as jb
import japanese_corpus_v1 as corpus
import japanese_event_v1 as jevent
import reading_curriculum_v1 as curriculum
import reading_comprehension_v1 as comprehension
import japanese_retell_v1 as retell
import japanese_sequence_v1 as sequence
import japanese_word_meaning_v1 as word_meaning
import caregiver_v1 as caregiver
import reading_llm_v1 as reading_llm
import cognition_v1 as cognition
import capability_probe_v1 as capability_probe
import japanese_dialogue_v1 as ja_dialogue
import japanese_prediction_v1 as ja_prediction
import semantic_representation_v1 as semantic_representation

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME = ROOT / ".local"
CURRICULUM_FILE = "reading-curriculum.json"
EVENTS_FILE = "reading-events.json"
COMPREHENSION_FILE = "reading-comprehension.json"
RETELL_FILE = "reading-retelling.json"
SEQUENCE_FILE = "reading-sequence.json"                 # general character model
NARRATIVE_SEQUENCE_FILE = "reading-narrative-sequence.json"   # P1-1: retell-only model
WORD_MEANING_FILE = "reading-word-meaning.json"
COGNITION_FILE = "cognition-state.json"
SEMANTIC_FILE = "semantic-representation.json"
PROBE_FILE = "cognition-probe.json"
JA_DIALOGUE_FILE = "japanese-dialogue.json"
PREDICTION_FILE = "reading-prediction.json"
CAREGIVER_FILE = "caregiver.json"
AIDED_FILE = "reading-aided.json"          # evidence-0 aided readings (diagnostic store)
STATUS_FILE = "reading-status.json"
SEQUENCE_TRAIN_SECONDS = 5.0
STOP_FILE = "READING_STOP"
SHELF_LOW_WATER = 4           # in-rotation books below this -> fetch more
FETCH_BUDGET = 20           # network requests per shelf-widening pass
FETCH_TARGET = 10           # books to add per pass
FETCH_COOLDOWN = 4           # cycles to wait between shelf-widening fetches
AOZORA_LEVEL_MARGIN = 1.0   # skip Aozora works more than this above the reading
                            # level -- padding the shelf with books the reader
                            # cannot handle is what stalled it (was 3.0)
TATOEBA_MAX_LEVEL = 4.0     # above this the parser handles literary prose well
                            # enough that sentence drills add little
TATOEBA_READERS_PER_LEVEL = 6   # once this many exist near the level, stop
VOCAB_FUEL_COOLDOWN = 5      # cycles between word-meaning fuel top-ups
VOCAB_FUEL_FRESH = 6        # fresh Tatoeba readers pulled per top-up
VOCAB_FUEL_BUFFER = 24     # fuel reader texts kept in the rolling buffer
VOCAB_FUEL_LEVELS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5)   # a wide concrete-noun span
VOCAB_FUEL_BAND = 0.7
                            # fetching so the band can drain and the level rise
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


HEURISTIC_SELF = "heuristic_self"


def _events_of(text: str) -> list[dict]:
    """Noise's OWN parse of a text: particle-anchored heuristic events, no
    morphological analyser, no LLM.  Every event is stamped `heuristic_self` so
    downstream code can prove its provenance before learning from it."""
    out = []
    for e in jevent.extract_story(text):
        d = dict(e.__dict__)
        d["provenance"] = HEURISTIC_SELF
        out.append(d)
    return out


def _heuristic_only(events: list[dict]) -> list[dict]:
    """Fail-closed: only an EXPLICIT `heuristic_self` stamp is Noise's own
    experience.  Missing / unknown / teacher / LLM / caregiver provenance is
    dropped -- a future analyser that forgets to stamp cannot leak into learning.
    """
    return [e for e in events if isinstance(e, dict) and e.get("provenance") == HEURISTIC_SELF]


def _quarantine_count(events_store: dict) -> int:
    return sum(1 for evs in events_store.values() for e in evs
              if not (isinstance(e, dict) and e.get("provenance") == HEURISTIC_SELF))


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


def _rnn_corpus(cur: dict, forbidden_cols: set[str]) -> dict[str, str]:
    """The GENERAL character model's training text: the RAW published text of
    every book Noise has READ -- including books the heuristic parser could not
    break into events (`shelved_stuck`) and the Tatoeba sentence bundles.  The
    model needs characters, not a parse, and gating its corpus on parser success
    is why it sat at 723K chars while 169 read books went unused.  Raw text is
    reading *input*, not an interpretation, so the `heuristic_self` provenance
    filter does not apply here.  What DOES still apply: every collection feeding a
    comprehension / retelling SELECTION / FINAL / RESERVE snapshot is kept out.

    This model is DIAGNOSTIC ONLY (character bits/char) and drives the display
    retellings.  It is NEVER the retelling benchmark's likelihood model -- its
    corpus is a superset of the narrative story set, so the position baseline
    built from "exactly the RNN's training sources" could never match it.  The
    dedicated narrative model (`_narrative_rnn_corpus`) exists for that."""
    texts: dict[str, str] = {}
    for book in cur.get("shelf", {}).values():
        text = book.get("text") or ""
        if not text or not curriculum.book_was_read(book):
            continue
        if jb.collection(book["url"]) in forbidden_cols:
            continue
        texts[book["url"]] = texts.get(book["url"], "") + text
    return texts


def _narrative_rnn_corpus(cur: dict, narrative_urls: set[str],
                          forbidden_cols: set[str]) -> dict[str, str]:
    """P1-1: the DEDICATED narrative model's training text.  RAW text, but ONLY
    of books that were recognised as narratives this cycle -- `narrative_urls` is
    exactly the url set of `all_stories` (read, >= 3 heuristic events, `source`
    != 'tatoeba').  Vocab-fuel bundles have no shelf entry so they cannot appear
    here; Tatoeba books are on the shelf but never in `narrative_urls`.  Every
    forbidden (selection / final / reserve / non-train tier) collection is
    excluded, so the model's training source set is disjoint from every retell
    test snapshot AND identical to the position baseline's source set."""
    texts: dict[str, str] = {}
    for book in cur.get("shelf", {}).values():
        url = book.get("url")
        text = book.get("text") or ""
        if not url or url not in narrative_urls or not text:
            continue
        if not curriculum.book_was_read(book):
            continue
        if jb.collection(url) in forbidden_cols:
            continue
        texts[url] = texts.get(url, "") + text
    return texts


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

    # Tatoeba graded readers: bundles of short, clean, single-clause sentences.
    # The Aozora/wikisource corpus is literary prose the parser mangles; Tatoeba
    # sentences it can actually read, so they build vocabulary and let the level
    # rise on a solid footing.  Pulled only near picture-book level and only when
    # the narrative sources came up short.
    tatoeba = []
    near_tatoeba = sum(1 for b in cur["shelf"].values()
                       if b.get("source") == "tatoeba"
                       and abs(b.get("estimated_level", 0) - level) <= 0.75)
    if (len(fetched) < FETCH_TARGET and budget_left() and level <= TATOEBA_MAX_LEVEL
            and near_tatoeba < TATOEBA_READERS_PER_LEVEL):
        skip = cur.get("_tatoeba_cursor", 0)
        want = min(FETCH_TARGET - len(fetched), TATOEBA_READERS_PER_LEVEL - near_tatoeba)
        try:
            tatoeba = corpus.tatoeba_readers(round(level, 1), n_readers=want,
                                             skip=skip, network=4)
        except Exception:
            tatoeba = []
        tatoeba = [t for t in tatoeba if t.url not in have]
        cur["_tatoeba_cursor"] = skip + len(tatoeba) * 20

    books = []
    for s in fetched:
        evs = _events_of(s.text)                  # parsed here only to estimate difficulty
        books.append({"title": s.title, "url": s.url, "source": "ja", "text": s.text,
                      "event_count": len(evs), "events": evs,
                      "verbs": [e.get("verb", "") for e in evs]})
    for s in tatoeba:
        evs = _events_of(s.text)
        books.append({"title": s.title, "url": s.url, "source": "tatoeba", "text": s.text,
                      "license": corpus.TATOEBA_LICENSE, "event_count": len(evs),
                      "events": evs, "verbs": [e.get("verb", "") for e in evs]})
    added = curriculum.register_books(cur, books, cycle)
    # NOTE: a fetched-but-unread book does NOT get its events written to the
    # shared events store.  Its events enter the store only when Noise actually
    # reads it (select_next_book -> record_reading), so an unread book can never
    # be training data, RNN text, a frozen-benchmark story, or vocabulary
    # evidence.  This is the read/unread boundary the re-audit requires.
    return added


def _refresh_vocab_fuel(cur: dict, cycle: int) -> int:
    """Keep a small rolling buffer of fresh Tatoeba reader texts for word
    meaning -- the corpus's only *renewable* source of concrete common nouns,
    which the finite Aozora shelf cannot give.

    The buffer feeds `japanese_word_meaning` ONLY (see `wm_stories`): the texts
    never enter the shelf, `events_store`, graduation, the RNN or any narrative
    benchmark.  Reading a sentence for its vocabulary is still Noise's own
    reading -- the events are parsed here and stamped `heuristic_self` -- but it
    is not a curriculum "reading", so it cannot move the level or a benchmark.
    Runs on its own cooldown; rotates Tatoeba level pools with per-pool cursors
    so the material is always new, wrapping a pool when it runs out."""
    if cycle - cur.get("_vocab_fuel_cycle", -999) < VOCAB_FUEL_COOLDOWN:
        return 0
    cur["_vocab_fuel_cycle"] = cycle
    idx = cur.get("_vocab_fuel_level_idx", 0)
    cursors = cur.setdefault("_vocab_fuel_cursors", {})
    buf = cur.get("_vocab_fuel_texts", [])
    seen = {t[:40] for t in buf}
    fresh = []
    for step in range(len(VOCAB_FUEL_LEVELS)):
        lvl = VOCAB_FUEL_LEVELS[(idx + step) % len(VOCAB_FUEL_LEVELS)]
        key = f"{lvl:.1f}"
        skip = cursors.get(key, 0)
        got = []
        for attempt_skip in (skip, 0) if skip else (skip,):
            try:
                got = corpus.tatoeba_readers(lvl, n_readers=VOCAB_FUEL_FRESH,
                                             per_reader=20, band=VOCAB_FUEL_BAND,
                                             skip=attempt_skip, network=3)
            except Exception:
                got = []
            if got:
                cursors[key] = (attempt_skip or 0) + len(got) * 20
                break
            cursors[key] = 0
        fresh = [r.text for r in got if r.text[:40] not in seen]
        if fresh:
            cur["_vocab_fuel_level_idx"] = (idx + step + 1) % len(VOCAB_FUEL_LEVELS)
            break
    if not fresh:
        return 0
    cur["_vocab_fuel_texts"] = (buf + fresh)[-VOCAB_FUEL_BUFFER:]
    return len(fresh)


_events_path: Path | None = None


def _read_events() -> dict:
    return _read(_events_path)


def _write_events(value: dict) -> None:
    # fail-closed at the store boundary: a book keeps only its heuristic_self
    # events, and a book left with < 3 of them is dropped entirely.
    clean = {}
    for bid, evs in value.items():
        keep = [e for e in evs if isinstance(e, dict) and e.get("provenance") == HEURISTIC_SELF]
        if len(keep) >= 3:
            clean[bid] = keep
    _write(_events_path, clean)


def run_once(runtime: Path) -> dict:
    global _events_path
    _events_path = runtime / EVENTS_FILE
    cur = _read(runtime / CURRICULUM_FILE) or curriculum.empty_curriculum()
    prev_comp = _read(runtime / COMPREHENSION_FILE)
    prev_retell = _read(runtime / RETELL_FILE)
    prev_seq = _read(runtime / SEQUENCE_FILE)
    prev_narr_seq = _read(runtime / NARRATIVE_SEQUENCE_FILE)
    cycle = cur.get("cycle", 0) + 1
    cur["cycle"] = cycle
    events_store = _read_events()

    # a better parser deserves a fresh reading of the whole shelf: drop the event
    # cache, put books it had set aside back in rotation, and re-walk from the
    # easiest level.
    from japanese_event_v1 import PARSER_VERSION
    CORPUS_VERSION = 5                            # bump when _modernise / cleaning changes
    if cur.get("corpus_version", 0) < CORPUS_VERSION:
        for b in cur["shelf"].values():
            if b.get("text"):
                b["text"] = corpus._modernise(corpus._strip_section_headings(b["text"]))
        cur["corpus_version"] = CORPUS_VERSION
        events_store = {}
    if cur.get("events_parser_version", 0) < PARSER_VERSION:
        events_store = {}
        cur["events_parser_version"] = PARSER_VERSION
    for bid in curriculum.reevaluate_stale_parses(cur):
        events_store.pop(bid, None)
    curriculum.reset_level_for_new_parser(cur, cycle)
    curriculum.recompute_level_from_stories(cur, cycle)

    # one-off, idempotent: rebuild difficulty + known_words from heuristic events
    # and return any aid-graduated book for re-evaluation
    migration = curriculum.migrate_reading_state(
        cur, events_store, extract=lambda t: _events_of(corpus._modernise(t)))
    if migration.get("migrated"):
        cur["events_parser_version"] = PARSER_VERSION

    # invariant, enforced every cycle: the shared events store holds ONLY the
    # heuristic parse of books Noise has actually read.  A fetched-but-unread
    # book, or one whose read markers were cleared, is dropped -- it can never be
    # training data, RNN text, a frozen-benchmark story, or vocabulary evidence.
    _read_now = {bid for bid, b in cur["shelf"].items() if curriculum.book_was_read(b)}
    _pruned = [bid for bid in events_store if bid not in _read_now]
    for bid in _pruned:
        events_store.pop(bid, None)
    if _pruned or migration.get("migrated"):
        _write_events(events_store)

    in_rotation = sum(1 for b in cur["shelf"].values() if b["status"] == "in_rotation")
    reachable = in_rotation + sum(1 for b in cur["shelf"].values()
                                 if b["status"] in ("shelved_above_level", "shelved_stuck"))
    fetched = 0
    cooled = cycle - cur.get("last_fetch_cycle", -FETCH_COOLDOWN) >= FETCH_COOLDOWN
    # honour the cooldown normally, but never sit with an empty shelf
    if (in_rotation < SHELF_LOW_WATER and cooled) or reachable == 0:
        fetched = _fetch_more_books(cur, cycle)
        cur["last_fetch_cycle"] = cycle
        # NB: _fetch_more_books no longer writes events (unread books stay out of
        # the store), so do NOT reload events_store here -- that would discard the
        # in-memory changes migrate_reading_state just made this cycle.
    # renewable concrete-noun fuel for word meaning, independent of the narrative
    # shelf: the Aozora shelf is finite and, once read, word meaning stops
    # growing.  This only tops up a text buffer consumed by `wm_stories` below.
    _refresh_vocab_fuel(cur, cycle)

    book_id = curriculum.retention_check_due(cur, cycle) or curriculum.select_next_book(cur)
    reading = {"status": "no_book"}
    scaffold = None
    model = None                                  # caregiver batch reads this even with no book
    aided_record = None
    if book_id:
        book = cur["shelf"][book_id]
        # the comprehension model trains on the HEURISTIC (`heuristic_self`) parse
        # of other books Noise has actually read -- never the book it scores.  A
        # book is NOT excluded just because an aided reading of it was once shown:
        # its own heuristic events are legitimate experience.  Only teacher / LLM
        # events are filtered out, and those never enter events_store anyway.
        others = [_heuristic_only(ev) for bid, ev in events_store.items()
                  if bid != book_id and cur["shelf"].get(bid, {}).get("times_read", 0) > 0]
        others = [ev for ev in others if len(ev) >= 3]
        model = comprehension.ComprehensionModel().fit(others) if len(others) >= 5 else None
        heuristic = _heuristic_only(events_store.get(book_id) or _events_of(book.get("text", "")))
        events_store.setdefault(book_id, heuristic)      # heuristic parse, always

        # ---- Noise's own reading: heuristic events ONLY drive comprehension,
        # graduation, level, schema, vocabulary, the model, the RNN and the
        # frozen benchmarks (ARCHITECTURE.md invariants 1, 10, 16) ----
        reading = curriculum.record_reading(cur, book_id, heuristic, cycle, model=model)
        reading["title"] = book["title"]
        reading["level"] = book["estimated_level"]
        retold = retell.retell(heuristic, max_sentences=10)
        reading["retelling"] = retold
        reading["retelling_score"] = retell.score_retelling(heuristic, retold)

        # ---- aided reading: morphological teacher + (if still too thin) the
        # local-LLM scaffold.  DIAGNOSTIC ONLY -- shown to the caregiver, never
        # fed back into any of the above ----
        aided = jevent.refine_event_dicts(heuristic)
        aided_source = "teacher" if aided != heuristic else "none"
        tried = cur.setdefault("llm_scaffolded", {})
        if (sum(1 for e in aided if e.get("verb")) < reading_llm.MIN_EVENTS
                and book_id not in tried and book.get("text")
                and reading_llm.OllamaReader().available()):
            base_known = {w for w, info in cur.get("known_words", {}).items()
                          if info.get("books", 0) >= 1}
            scaffold = reading_llm.simplify_story(book["text"], base_known_words=base_known)
            tried[book_id] = {"status": scaffold["status"],
                              "event_count": scaffold.get("event_count", 0),
                              "content_kept": scaffold.get("content_kept"), "cycle": cycle}
            if scaffold.get("verified"):
                aided, aided_source = scaffold["events"], "llm_scaffold"
        if aided_source != "none" and len([e for e in aided if e.get("verb")]) >= 3:
            prov = "morphological_teacher" if aided_source == "teacher" else "local_llm_scaffold"
            # the aided reading as a whole is evidence 0 -- stamp every event with
            # the aid's provenance (overriding any inherited `heuristic_self`) so
            # it can never be mistaken for Noise's own parse downstream
            aided = [{**e, "provenance": prov} if isinstance(e, dict) else e for e in aided]
            # audit breadcrumb only -- NOT used to exclude the book's heuristic events
            book.setdefault("aided_shown_cycles", []).append(cycle)
            book["aided_ever"] = True                    # legacy audit flag, no longer gates learning
            aided_retold = retell.retell(aided, max_sentences=10)
            reading["aided"] = {
                "source": aided_source, "provenance": prov,
                "comprehension": (comprehension.book_comprehension(
                    aided, model, curriculum._known_set(cur))["score"] if model
                    else curriculum._self_consistency(aided)),
                "retelling": aided_retold,
                "retelling_score": retell.score_retelling(aided, aided_retold),
                "note": "evidence score 0 -- stored in reading-aided.json; does not "
                        "affect graduation, level, vocabulary, the model, the RNN, "
                        "or any frozen benchmark"}
            aided_record = {"cycle": cycle, "book_id": book_id, "title": book["title"],
                            "url": book["url"], **reading["aided"]}

    advance = curriculum.maybe_advance_level(cur, cycle)

    # frozen-benchmark capability measurements.  events_store holds only READ
    # books' `heuristic_self` events; filter by provenance again (fail-closed:
    # only an explicit `heuristic_self` stamp is Noise's own experience).
    real = set(events_store)
    heur_store = {bid: _heuristic_only(ev) for bid, ev in events_store.items()}
    # Tatoeba readers are sentence bundles, not narratives -- they feed
    # vocabulary and the RNN but never the narrative comprehension / retelling
    # benchmarks (no protagonist, no order to recover).
    all_stories = [{"url": cur["shelf"][bid]["url"], "events": ev}
                   for bid, ev in heur_store.items()
                   if bid in cur["shelf"] and len(ev) >= 3
                   and cur["shelf"][bid].get("source") != "tatoeba"]
    # word meaning is distributional, not narrative -- it DOES want the Tatoeba
    # sentence bundles: they are Noise's own reading (heuristic_self events) and
    # the corpus's richest source of concrete common nouns.  Graduated readers
    # are pruned from events_store on a parser bump and never re-selected, so
    # re-parse any short read book that is missing (its own heuristic parse of a
    # text it genuinely read is legitimate `heuristic_self` experience).
    wm_stories = [{"url": cur["shelf"][bid]["url"], "events": ev}
                  for bid, ev in heur_store.items()
                  if bid in cur["shelf"] and len(ev) >= 3]
    for bid, b in cur["shelf"].items():
        if (bid not in heur_store and curriculum.book_was_read(b)
                and b.get("text") and len(b["text"]) < 4000):
            evs = _heuristic_only(_events_of(corpus._modernise(b["text"])))
            if len(evs) >= 3:
                wm_stories.append({"url": b["url"], "events": evs})
    # renewable concrete-noun fuel: Tatoeba reader texts Noise parsed itself,
    # for word meaning only -- never a shelf book, never a benchmark story.
    from collections import Counter as _Counter
    _fuel_tok: _Counter = _Counter()
    for i, text in enumerate(cur.get("_vocab_fuel_texts", [])):
        evs = _heuristic_only(_events_of(corpus._modernise(text)))
        if len(evs) >= 3:
            wm_stories.append({"url": f"vocab-fuel://{i}", "events": evs})
            for e in evs:
                for slot in ("subject", "obj"):
                    tok = (e.get(slot) or "").strip()
                    if tok:
                        _fuel_tok[tok] += 1
    # fuel tokens seen >= 2 times join the word-meaning vocabulary (they never
    # become curriculum-"known" via a graduated book, but they are still words
    # Noise read and can research)
    wm_known = curriculum._known_set(cur) | {t for t, n in _fuel_tok.items() if n >= 2}
    narrative_urls = {s["url"] for s in all_stories}

    # cumulative training ledgers (collections a model has EVER trained on).  A
    # collection a model trained on is PINNED to the `train` tier of the
    # benchmark that TESTS that model, and can never be forced into its held-out
    # tier -- that pin protects earned weights when a regime is re-drawn.
    #   * comprehension tests its own n-gram model -> pins the GENERAL RNN ledger
    #     (pre-existing wiring, unchanged).
    #   * retelling tests the DEDICATED NARRATIVE RNN -> pins the narrative
    #     ledger only (P1-1).  The general RNN is NOT the retelling likelihood
    #     model and earns no retelling capability, so it does NOT have to avoid
    #     retelling's held-out collections; forcing it to was retiring its 150k
    #     earned steps every time the tiers moved.
    gen_ever_trained_cols = set(
        (prev_seq.get("training_data_fingerprint") or {}).get("ever_trained_collections") or [])
    narr_ever_trained_cols_prev = set(
        (prev_narr_seq.get("training_data_fingerprint") or {}).get("ever_trained_collections") or [])
    comp_forbidden = comprehension.forbidden_training_collections(
        prev_comp, all_stories, ever_trained_collections=gen_ever_trained_cols, cycle=cycle)
    retell_forbidden = retell.forbidden_training_collections(
        prev_retell, all_stories, ever_trained_collections=narr_ever_trained_cols_prev, cycle=cycle)
    # the general model avoids only the comprehension held-out set; the dedicated
    # narrative model avoids BOTH (comprehension's, so the two capability test
    # sets never overlap its corpus, and retelling's own).
    general_forbidden = comp_forbidden
    narrative_forbidden = comp_forbidden | retell_forbidden

    def _archive_retired_rnn(report: dict, label: str) -> None:
        retired = report.get("retired_model")
        if retired and retired.get("retired_state"):
            audit_dir = runtime / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            _write(audit_dir / f"{label}-retired-{stamp}.json", retired)
            report["retired_model"] = {k: v for k, v in retired.items() if k != "retired_state"}

    # (1) GENERAL character model -- raw text of every read book (Tatoeba, and
    # books the parser could not break down, included).  Diagnostic bits/char +
    # the display retellings.  NEVER the retelling benchmark's likelihood model.
    seq_texts = _rnn_corpus(cur, general_forbidden)
    seq_report = sequence.train_and_evaluate(
        seq_texts, prev_seq, SEQUENCE_TRAIN_SECONDS, regime=sequence.TRAINING_REGIME,
        training_context={
            "training_regime": sequence.TRAINING_REGIME, "parser_version": PARSER_VERSION,
            "provenance_policy": curriculum.PROVENANCE_POLICY, "read_only": True,
            "forbidden_collections": sorted(general_forbidden)})
    _archive_retired_rnn(seq_report, "reading-sequence")
    gen_ever_trained_cols = set(
        (seq_report.get("training_data_fingerprint") or {}).get("ever_trained_collections") or [])

    # (2) DEDICATED NARRATIVE model -- raw text of recognised-narrative books
    # ONLY, minus every forbidden (selection/final/reserve/non-train) collection.
    # Its training source set is BY CONSTRUCTION identical to the retelling
    # position baseline's source set, so the P1-6 corpus-match check can pass.
    narr_texts = _narrative_rnn_corpus(cur, narrative_urls, narrative_forbidden)
    narr_report = sequence.train_and_evaluate(
        narr_texts, prev_narr_seq, SEQUENCE_TRAIN_SECONDS, regime=sequence.NARRATIVE_REGIME,
        training_context={
            "training_regime": sequence.NARRATIVE_REGIME, "parser_version": PARSER_VERSION,
            "provenance_policy": curriculum.PROVENANCE_POLICY, "read_only": True,
            "forbidden_collections": sorted(narrative_forbidden)})
    _archive_retired_rnn(narr_report, "reading-narrative-sequence")
    narr_ever_trained_cols = set(
        (narr_report.get("training_data_fingerprint") or {}).get("ever_trained_collections") or [])
    narr_training_urls = sorted(
        s["url"] for s in (narr_report.get("training_data_fingerprint") or {}).get("ever_trained_sources", []))

    comp_report = comprehension.evaluate_comprehension(
        all_stories, prev_comp, ever_trained_collections=gen_ever_trained_cols, cycle=cycle)

    # retelling -- one-time archival of the pre-v5 report (measured against the
    # general RNN and therefore permanently measurement_invalid), then the
    # benchmark re-runs on the dedicated narrative model whose training URLs ARE
    # the position baseline's source set.
    if prev_retell.get("eval_regime") and prev_retell.get("eval_regime") != retell.EVAL_REGIME:
        audit_dir = runtime / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        _write(audit_dir / f"reading-retelling-retired-{stamp}.json", {
            "retired_at": stamp,
            "reason": "P1-1: retelling now uses a dedicated narrative RNN; the prior "
                      "regime scored against the general character RNN whose corpus "
                      "(Tatoeba bundles + unparseable books) is a permanent superset of "
                      "the narrative story set -> measurement_invalid_baseline_corpus_mismatch. "
                      "No capability was ever confirmed under the retired regime.",
            "retired_eval_regime": prev_retell.get("eval_regime"),
            "new_eval_regime": retell.EVAL_REGIME,
            "retired_report": prev_retell})
    retell_report = retell.evaluate_retelling(
        all_stories, prev_retell, rnn_state=narr_report.get("state"),
        rnn_training_urls=narr_training_urls,
        ever_trained_collections=narr_ever_trained_cols, cycle=cycle)

    # word meaning -- learn what the entities Noise reads DENOTE, from
    # ja.wiktionary / ja.wikipedia (CC-BY-SA); tested by self-explanation
    prev_wm = _read(runtime / WORD_MEANING_FILE)
    # last cycle's prediction-failure feedback: capped counter-evidence to the
    # genus a word's story behaviour kept contradicting (one channel, invariant 10)
    pred_feedback = (_read(runtime / PREDICTION_FILE) or {}).get("prediction_feedback", {})
    try:
        wm_report = word_meaning.learn_and_evaluate(
            wm_stories, prev_wm, cycle, known_words=wm_known,
            prediction_feedback=pred_feedback)
    except Exception as exc:                      # never let it break the reader
        wm_report = {**(prev_wm or {}), "status": "error", "error": repr(exc)}
    _write(runtime / WORD_MEANING_FILE, wm_report)

    # A compact predictive representation learned only from Noise's own parsed
    # events.  It supplies hypotheses to cognition; dictionary/LLM labels are
    # never training targets and the diagnostic below never grants capability.
    semantic_report = _read(runtime / SEMANTIC_FILE)
    if semantic_representation.enabled():
        try:
            semantic_report = semantic_representation.learn(
                heur_store, wm_report, semantic_report, PARSER_VERSION, cycle)
        except Exception as exc:
            semantic_report = {**(semantic_report or {}), "status": "error", "error": repr(exc)}
        _write(runtime / SEMANTIC_FILE, semantic_report)

    # knowledge -> capability loop: retrieve -> abstract -> generate a problem ->
    # reason -> self-evaluate -> store the attempt.  Measured by a FROZEN probe,
    # not by rule count.  AI_NOISE_COGNITION=0 turns it off.
    cog_report = _read(runtime / COGNITION_FILE)
    probe_report = _read(runtime / PROBE_FILE)
    if (probe_report.get("version") and probe_report.get("version") != capability_probe.VERSION):
        audit_dir = runtime / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        _write(audit_dir / f"cognition-probe-retired-{stamp}.json", {
            "retired_at": stamp, "retired_version": probe_report.get("version"),
            "new_version": capability_probe.VERSION,
            "reason": "P1-3: baseline post-selection.  The retired probe kept ONLY "
                      "problems the frequency baseline failed, so its `lift` was the "
                      "raw derive-rate, not evidence of beating a general baseline.  "
                      "Its history is NOT reinterpreted as a fair final result; the "
                      "surviving baseline-fails problems are re-used as the new "
                      "CHALLENGE tier only.",
            "retired_report": probe_report})
    if cognition.enabled():
        try:
            cog_report = cognition.run_cognitive_cycle(
                cycle=cycle, wm_state=wm_report, heur_store=heur_store,
                shelf=cur["shelf"], just_read=book_id, previous=cog_report,
                semantic_state=semantic_report)
            probe_report = capability_probe.maybe_run(
                cycle, wm_report, heur_store, cur["shelf"], cog_report, probe_report,
                semantic_state=semantic_report)
        except Exception as exc:                  # isolate: a failure never stalls reading
            cog_report = {**(cog_report or {}), "status": "error", "error": repr(exc)}
        _write(runtime / COGNITION_FILE, cog_report)
        _write(runtime / PROBE_FILE, probe_report)

    # predict the next event's verb CLASS from Noise's concepts, grade it,
    # revise a concept on a persistent miss.  The self-correction loop the
    # Japanese side was missing.  AI_NOISE_JA_PREDICTION=0 turns it off.
    pred_report = _read(runtime / PREDICTION_FILE)
    if ja_prediction.enabled():
        try:
            pred_report = ja_prediction.run_prediction(
                cycle, all_stories, heur_store.get(book_id, []) if book_id else [],
                wm_report.get("beliefs", {}), (cog_report or {}).get("rules", []), pred_report)
        except Exception as exc:                      # isolate: never stalls reading
            pred_report = {**(pred_report or {}), "status": "error", "error": repr(exc)}
        _write(runtime / PREDICTION_FILE, pred_report)

    # say something in Japanese from what reading taught, learn if it landed.
    # A use-test for the beliefs; the partner is an environment, not a teacher.
    # AI_NOISE_JA_DIALOGUE=0 turns it off.
    dlg_report = _read(runtime / JA_DIALOGUE_FILE)
    if (dlg_report.get("version") and dlg_report.get("version") != ja_dialogue.VERSION):
        audit_dir = runtime / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        _write(audit_dir / f"japanese-dialogue-retired-{stamp}.json", {
            "retired_at": stamp, "retired_version": dlg_report.get("version"),
            "new_version": ja_dialogue.VERSION,
            "reason": "P1-2: echo-aware scoring.  The prior scoring counted a "
                      "verbatim/partial echo or a helpful guess at a malformed "
                      "utterance as 'understood'; those understood rates are NOT "
                      "carried into the new regime.",
            "retired_report": dlg_report})
    if ja_dialogue.enabled():
        try:
            dlg_report = ja_dialogue.run_practice(
                wm_report, (cog_report or {}).get("rules", []), dlg_report, cycle)
        except Exception as exc:
            dlg_report = {**(dlg_report or {}), "status": "error", "error": repr(exc)}
        _write(runtime / JA_DIALOGUE_FILE, dlg_report)

    # a free-generation retelling of the book just read, conditioned ONLY on the
    # event representation Noise formed (no gold text, no LLM rephrasing).
    # DISPLAY ONLY, earns no capability credit -- but generated from the DEDICATED
    # narrative model (disjoint from every retell test snapshot by construction),
    # not the general model whose corpus can overlap the retell selection set.
    _free_state = narr_report.get("state") or seq_report.get("state")
    if book_id and _free_state and heuristic:
        free = retell.free_retell(_free_state, retell._event_repr(heuristic))
        if free:
            reading["free_retelling"] = free
            reading["free_retelling_score"] = retell.score_retelling(heuristic, free)

    # low-effort caregiver check: a small batch of multiple-choice / yes-no
    # questions about recently-read books, every CAREGIVER_INTERVAL cycles
    care_state = _read(runtime / CAREGIVER_FILE) or caregiver.empty_state()
    caregiver.expire_if_stale(care_state, cycle)
    if caregiver.due(care_state, cycle):
        current_fidelity = (reading.get("retelling_score") or {}).get("fidelity")
        recent = [{"title": b["title"], "url": b["url"],
                   # the caregiver judges Noise's OWN reading (heuristic events),
                   # the same one record_reading scored
                   "events": heur_store.get(bid, []),
                   "fidelity": current_fidelity if bid == book_id else None,
                   "_last": b.get("last_read_cycle") or 0}
                  for bid, b in cur["shelf"].items()
                  if b.get("times_read", 0) > 0 and len(heur_store.get(bid, [])) >= 3]
        recent.sort(key=lambda r: -r["_last"])
        questions = caregiver.generate_questions(recent, cycle, model)
        if questions:
            caregiver.open_batch(care_state, questions, cycle)
    _write(runtime / CAREGIVER_FILE, care_state)

    # evidence-0 aided readings live in their OWN store, keyed by cycle, never
    # merged into reading-events.json
    if aided_record is not None:
        aided_store = _read(runtime / AIDED_FILE) or {"readings": [], "note":
            "evidence score 0 -- morphological-teacher / local-LLM readings shown "
            "to the caregiver; never used for learning, evaluation or graduation"}
        aided_store["readings"] = (aided_store.get("readings", []) + [aided_record])[-300:]
        _write(runtime / AIDED_FILE, aided_store)
    aided_total = len((_read(runtime / AIDED_FILE) or {}).get("readings", []))

    _write(runtime / CURRICULUM_FILE, cur)
    _write_events(events_store)
    _write(runtime / COMPREHENSION_FILE, comp_report)
    _write(runtime / RETELL_FILE, retell_report)
    _write(runtime / SEQUENCE_FILE, seq_report)
    _write(runtime / NARRATIVE_SEQUENCE_FILE, narr_report)
    status = {
        "heartbeat": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cycle": cycle, "books_fetched": fetched,
        "reading": reading, "level_advance": advance,
        "curriculum": curriculum.summary(cur),
        "word_meaning": {k: wm_report.get(k) for k in
                         ("status", "vocab", "researched_count", "taxonomy_size",
                          "belief_count", "understood_count", "understood_rate",
                          "corrections", "llm_status", "llm_asked_count",
                          "test_words", "measured", "mean_gain", "z",
                          "significant_now", "capability_confirmed",
                          "sample_explanations", "prediction_feedback_applied")},
        "semantic_representation": {k: (semantic_report or {}).get(k) for k in
                         ("status", "version", "parser_version", "word_count",
                          "context_count", "pairs_seen", "sources_trained_this_cycle",
                          "positive_pairs_this_cycle", "pending_sources",
                          "model_fingerprint", "diagnostic", "reset_reason", "note", "error")},
        "cognition": {k: (cog_report or {}).get(k) for k in
                      ("status", "concept", "rules_total", "rules_reusable",
                       "experiences_total", "problems_this_cycle", "correct_this_cycle",
                       "live_solve_rate", "repeated_failure_rate", "by_level",
                       "controller_policy", "controller_decisions", "sample_experience")},
        "cognition_probe": {k: (probe_report or {}).get(k) for k in
                            ("status", "frozen_at", "have", "need",
                             "tier_separation_valid", "tier_sizes",
                             "challenge_problem_count", "challenge_first_derive_rate",
                             "challenge_latest_derive_rate", "challenge_before_after_gain",
                             "challenge_trend", "challenge_note",
                             "selection_problem_count", "selection_measurements",
                             "selection_latest", "selection_models_clearing_threshold",
                             "selection_pass", "selection_trend",
                             "final_problem_count", "reserve_problem_count",
                             "final_opened_count", "final_result",
                             "capability_confirmed", "capability_basis",
                             "capability_pending_reason", "migrated_from_version",
                             "reference_pool_by_genus", "pairable_genera", "bottleneck",
                             "challenge_have", "final_have",
                             # legacy keys (challenge-backed) for status continuity
                             "problem_count", "first_derive_rate", "latest_derive_rate",
                             "latest_lift", "before_after_gain", "trend", "latest_by_level")},
        "prediction": {k: (pred_report or {}).get(k) for k in
                       ("status", "eval_regime", "regime_reset_from", "train_stories",
                        "recomputed", "selection", "selection_significant_streak",
                        "selection_next_streak_train", "selection_measurements",
                        "selection_stories", "reserve_stories", "final_status",
                        "final_opened_count", "final_query_budget", "final_result",
                        "capability_confirmed", "capability_confirmed_ever",
                        "capability_confirmed_current_model", "confirmed_checkpoints",
                        "capability_pending_reason", "prediction_trend",
                        "secondary_diagnostic_next_verb_class",
                        "live_this_cycle", "outcomes_tracked", "feedback_words",
                        "corrupters", "collections_disjoint", "tier_collection_counts")},
        "japanese_dialogue": {k: (dlg_report or {}).get(k) for k in
                              ("status", "frozen_count", "practice_turns", "probe_rounds",
                               "best_strategy", "first_understood_rate", "overall_understood_rate",
                               "before_after_gain", "trend", "sample_turn",
                               "recent_echo_rate", "recent_clarification_rate",
                               "recent_new_info_rate", "recent_requested_act_rate",
                               "recent_malformed_rate", "recent_belief_supported_rate",
                               "latest_round_metrics", "have", "need")},
        "comprehension": {k: comp_report.get(k) for k in
                          ("status", "comprehension_score", "consequence",
                           "consequence_baseline", "consequence_z", "beats_baseline",
                           "capability_confirmed", "capability_confirmed_ever",
                           "capability_confirmed_current_model", "capability_pending_reason",
                           "confirmed_checkpoints", "comprehension_trend", "test_stories",
                           "eval_regime", "regime_reset_from", "snapshot_migrated",
                           "selection_stories", "selection_fingerprint", "selection_measurements",
                           "selection_significant_streak", "selection_next_streak_train",
                           "reserve_stories", "final_status", "final_opened_count",
                           "final_query_budget", "final_result",
                           "tier_collection_counts", "tier_collections_ever_trained",
                           "topup_short_tiers", "selection_insufficient_reason",
                           "collection_disjointness", "collections_disjoint")},
        "retelling": {k: retell_report.get(k) for k in
                      ("status", "roundtrip_fidelity", "order_gain", "gain_z",
                       "rnn_pairwise_accuracy", "position_baseline_accuracy",
                       "beats_baseline", "capability_confirmed", "capability_confirmed_ever",
                       "capability_confirmed_current_model", "capability_pending_reason",
                       "confirmed_checkpoints", "retelling_trend", "test_stories", "eval_regime",
                       "regime_reset_from", "snapshot_migrated", "recomputed",
                       "selection_stories", "selection_fingerprint", "selection_measurements",
                       "selection_significant_streak", "selection_next_streak_train",
                       "reserve_stories", "final_status", "final_opened_count",
                       "final_query_budget", "final_result",
                       "tier_collection_counts", "tier_collections_ever_trained",
                       "collection_disjointness", "collections_disjoint",
                       "baseline_corpus_matches_rnn", "baseline_source_fingerprint",
                       "baseline_source_count", "rnn_training_url_count",
                       "rnn_training_urls_missing_from_corpus",
                       "rnn_fingerprint", "rnn_training_regime", "rnn_steps_at_eval",
                       "rnn_steps_now", "rnn_model_changed", "next_reeval")},
        "self_vs_aided": {
            "self_comprehension": (reading.get("comprehension")
                                   if isinstance(reading, dict) else None),
            "aided_comprehension": (reading.get("aided") or {}).get("comprehension")
                                   if isinstance(reading, dict) else None,
            "aided_source": (reading.get("aided") or {}).get("source")
                            if isinstance(reading, dict) else None,
            "note": "aided は証拠スコア0（卒業・レベル・語彙・固定検証に不算入）"},
        "sequence": {k: seq_report.get(k) for k in
                     ("status", "capability_status", "held_out_bits_per_char",
                      "baseline_bits_per_char", "improvement_bits", "improvement_z",
                      "improvement_significant_now", "perplexity_trend", "steps_trained",
                      "model_fingerprint", "training_regime", "split_policy_version",
                      "contamination_status", "reset_reason", "reset_collisions",
                      "boundary_migrated", "started_clean_at", "parent_model_fingerprint",
                      "ever_trained_source_count", "sources_added_this_cycle",
                      "dropped_trained_sources", "significant_streak")},
        "sequence_ever_trained_fingerprint":
            (seq_report.get("training_data_fingerprint") or {}).get("ever_trained_set_fingerprint"),
        "sequence_boundary_fingerprint":
            (seq_report.get("training_data_fingerprint") or {}).get("boundary_fingerprint"),
        "sequence_ever_trained_collections":
            len((seq_report.get("training_data_fingerprint") or {}).get("ever_trained_collections") or []),
        "sequence_retirement_log": seq_report.get("retirement_log", []),
        "sequence_sample": (seq_report.get("samples") or [""])[0],
        # P1-1: the dedicated narrative model + how it lines up with the retelling
        # benchmark's position baseline (they MUST share a training source set)
        "narrative_sequence": {k: narr_report.get(k) for k in
                     ("status", "capability_status", "held_out_bits_per_char",
                      "baseline_bits_per_char", "improvement_bits", "improvement_z",
                      "perplexity_trend", "steps_trained", "model_fingerprint",
                      "training_regime", "contamination_status", "reset_reason",
                      "ever_trained_source_count", "sources_added_this_cycle",
                      "train_chars", "held_out_chars")},
        "narrative_sequence_training": {
            "corpus_source_count": len(narr_texts),
            "ever_trained_source_count":
                (narr_report.get("training_data_fingerprint") or {}).get("ever_trained_source_count"),
            "ever_trained_collections":
                len((narr_report.get("training_data_fingerprint") or {}).get("ever_trained_collections") or []),
            "retell_baseline_source_count": retell_report.get("baseline_source_count"),
            "retell_corpus_matches_rnn": retell_report.get("baseline_corpus_matches_rnn"),
            "retell_eval_valid": retell_report.get("status") not in
                ("measurement_invalid", "measurement_invalid_baseline_corpus_mismatch"),
            "narrative_urls_recognised": len(narrative_urls),
            "retired_retell_regime": prev_retell.get("eval_regime")
                if (prev_retell.get("eval_regime") and prev_retell.get("eval_regime") != retell.EVAL_REGIME)
                else None},
        "caregiver": caregiver.summary(care_state),
        "caregiver_questions": [q["prompt"] for q in care_state.get("pending", [])],
        "llm_scaffold": ({"book": reading.get("title"), **{k: scaffold.get(k) for k in
                          ("status", "event_count", "content_kept", "length_ratio")}}
                         if scaffold else None),
        "llm_scaffold_totals": _scaffold_totals(cur.get("llm_scaffolded", {})),
        "aided_reading": reading.get("aided"),          # evidence 0, diagnostic only
        "aided_store": {"file": AIDED_FILE, "total_readings": aided_total,
                        "this_cycle": aided_record is not None},
        "provenance": {"policy": curriculum.PROVENANCE_POLICY,
                       "events_store_non_self_events": _quarantine_count(events_store),
                       "events_store_books": len(events_store)},
        "schema_migration": migration if migration.get("migrated") else
                            {"schema_version": cur.get("curriculum_schema_version")},
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
             f"帰結確率={comp.get('consequence')}/unigram{comp.get('consequence_baseline')} "
             f"z={comp.get('consequence_z')} 基準超え={comp.get('beats_baseline')} "
             f"傾向={comp.get('comprehension_trend')}",
             f"再話(固定) : 生成利得={ret.get('generation_gain')}（往復診断={ret.get('roundtrip_fidelity')}）"
             f" z={ret.get('gain_z')} 基準超え={ret.get('beats_baseline')} "
             f"傾向={ret.get('retelling_trend')} [{ret.get('status')}]",
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
