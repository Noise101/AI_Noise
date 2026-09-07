#!/usr/bin/env python3
"""Developmental reading curriculum: a bookshelf and a reading-level state machine.

Replaces the link-crawler ("read whatever is adjacent") with the vision:
read books a young child can read, reread until understood, advance level only
when comprehension is demonstrated, and never discard anything -- material above
the current level is shelved with a level tag and returns automatically when
the reader catches up.

  * `text_difficulty` scores a story (sentence length, kanji density, unknown-
    word ratio, events per sentence, subordinate-clause density) into an
    `estimated_level` (~1 = first picture books, ~5 = literary prose),
    calibrated on a handful of reference retellings.
  * `select_next_book` picks the in-rotation book closest to the reader's ZPD
    (known-word coverage ~90-97%), preferring the least-read.
  * `record_reading` updates known words, comprehension history and status:
    graduate on a passing comprehension score, shelve after MAX_REREADS with
    no gain, otherwise stay in rotation.
  * `maybe_advance_level` raises the level after sustained comprehension at the
    current band and un-shelves everything now within reach.  Never auto-demotes.
  * `retention_check_due` returns a graduated lower-level book to re-test, so a
    lying level model (catastrophic forgetting) is caught.

Vocabulary is tracked in three tiers, because a reference link is not
understanding (a 4-year-old knows a word long before it can define one):
  * seen      -- appeared in >= KNOWN_AFTER_BOOKS distinct books, parser handled it
  * used      -- demonstrated correct use AND rejected a wrong use, on held-out
                 contexts it was not taught in (behavioural understanding)
  * explained -- produced a description that itself predicts correct usage
                 (metalinguistic; the strict, aspirational bar -- a capstone)
Level and coverage count a word as "known" at the `used` tier once use-tests
exist, and fall back to `seen` during the cold start.

Nothing here calls the network or an LLM.  The comprehension score and the
per-word use/explain outcomes are supplied by the caller (reading_comprehension
_v1, next phase); a cheap self-consistency proxy is used for comprehension
until then.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

WORD = re.compile(r"[぀-ヿ㐀-鿿]+")
KANJI = re.compile(r"[㐀-䶿一-鿿]")
SENT = re.compile(r"[。！？]")
# Clause-linking connectives.  Anchored to a comma or clause break where
# possible: bare ば/たら/なら/たり match far too many word-internal substrings
# (のばす, ことば, あたり, ...) and made plain 敬体 folktales look like literary
# prose.
SUBORDINATE = re.compile(
    r"(ので|けれど|けれども|ながら|のに|ように|ため|という"
    r"|し、|が、|ば、|たら、|なら、|ても、|でも、|から、|と、)")

GRADUATE_COMPREHENSION = 0.75      # a book is "understood" at/above this
ADVANCE_COMPREHENSION = 0.80       # mean over recent graduates to raise the level
STALL_COMPREHENSION_DROP = 0.15    # recent comprehension this far below -> pause advancement
MAX_REREADS = 6                    # rereads before a stuck book is shelved
GRADUATES_TO_ADVANCE = 4           # graduated books at the band before advancing
LEVEL_STEP = 0.5
KNOWN_AFTER_BOOKS = 2              # a word is "known" after this many distinct books
ZPD_KNOWN_LOW, ZPD_KNOWN_HIGH = 0.88, 0.98
RETENTION_INTERVAL = 25           # cycles between retention re-tests

# The endpoint goal is middle-school reading (中学生); a picture-book level is
# only the FIRST milestone.  estimated_level scale (roughly):
#   ~1.5  絵本 / first reader          (~4歳)   -- first milestone
#   ~3.0  昔話再話 (楠山正雄 style)      (小1-2)
#   ~5.0  翻訳童話 / 新美南吉            (小5-6)
#   ~7.0  宮沢賢治 / 随筆                (中1-2)
#   ~9.0  芥川 / 説明文                  (中3+)   -- endpoint goal
MILESTONES = ((1.5, "絵本 (4歳・第一目標)"), (3.0, "小学校低学年"),
              (5.0, "小学校高学年"), (7.0, "中学生"), (9.0, "中学卒業レベル (最終目標)"))
MAX_LEVEL = 10.0


def explained_ratio_target(level: float) -> float:
    """How much of recently-learned vocabulary should reach the `explained`
    tier before advancing.  Zero for early readers (a 4-year-old defines
    nothing); ramps in from level 4 so that by the middle-school endpoint most
    words the reader claims are ones it can actually explain."""
    if level < 4.0:
        return 0.0
    return round(min(0.8, 0.15 * (level - 4.0)), 3)


def _book_id(url: str, title: str) -> str:
    return hashlib.sha256(f"{url}|{title}".encode()).hexdigest()[:16]


def _schema_signature(verbs: list[str]) -> list[str]:
    """A book's narrative schema, coarsely: the distinct verbs it uses.  Two
    stories that share many verbs tend to share structure (find -> want ->
    try -> fail ...), so the curriculum can present familiar structure first."""
    return sorted({v for v in verbs if v})[:40]


def familiar_schema(curriculum: dict) -> set[str]:
    """Verbs from every book the reader has already understood."""
    return {v for b in curriculum["shelf"].values()
            if b["status"] == "graduated" for v in b.get("schema", [])}


def text_difficulty(text: str, event_count: int, known_words: set[str],
                    events: "list[dict] | None" = None) -> dict:
    sentences = [s for s in SENT.split(text) if len(s) > 3]
    n_sent = max(1, len(sentences))
    total_chars = sum(len(s) for s in sentences)
    kanji = len(KANJI.findall(text))
    # `known_words` are event tokens (きつね, ぶどう, 見つける), so coverage must be
    # measured over the same units -- WORD.findall returns whole particle-glued
    # runs (「きつねがぶどうを見つけました」) that never match, forcing coverage to 0.
    if events:
        vocab = {t for e in events
                 for t in (e.get("subject"), e.get("obj"), e.get("verb")) if t}
    else:
        vocab = set(WORD.findall(text))
    unknown = [w for w in vocab if w not in known_words]
    features = {
        "sentences": len(sentences),
        "mean_sentence_chars": round(total_chars / n_sent, 1),
        "kanji_density": round(kanji / max(1, total_chars), 3),
        "distinct_words": len(vocab),
        "unknown_word_ratio": round(len(unknown) / max(1, len(vocab)), 3),
        "known_word_coverage": round(1 - len(unknown) / max(1, len(vocab)), 3),
        "events_per_sentence": round(event_count / n_sent, 2),
        "subordinate_density": round(len(SUBORDINATE.findall(text)) / n_sent, 2),
    }
    # Calibrated so a 敬体 folktale retelling (short sentences, light kanji) sits
    # near 1.5-2.0, a translated fairy tale near 3, literary prose (ごんぎつね)
    # near 4, 宮沢賢治 near 5.  Adjust the four coefficients, not add terms.
    level = (0.7
             + 0.030 * min(45, features["mean_sentence_chars"])
             + 4.0 * features["kanji_density"]
             + 0.9 * features["subordinate_density"])
    features["estimated_level"] = round(max(1.0, min(MAX_LEVEL, level)), 2)
    return features


def _self_consistency(events: list[dict]) -> float:
    """Cold-start comprehension proxy (used until reading_comprehension_v1 has
    enough read books to fit a model): a story is 'coherent' if one protagonist
    carries most events, there is a beginning-middle-end spread of distinct
    verbs, and the extracted structure reads as Japanese rather than as
    'それが<壊れた動詞>' on every clause."""
    if len(events) < 3:
        return 0.0
    from japanese_retell_v1 import retelling_coherence
    subjects = Counter(e.get("subject", "") for e in events)
    protagonist_share = subjects.most_common(1)[0][1] / len(events)
    distinct_verbs = len({e.get("verb", "") for e in events})
    verb_spread = min(1.0, distinct_verbs / max(3, len(events) * 0.5))
    mean_conf = sum(e.get("confidence", 0.0) for e in events) / len(events)
    coherence = retelling_coherence(events)
    return round(0.25 * protagonist_share + 0.2 * verb_spread
                 + 0.15 * mean_conf + 0.4 * coherence, 3)


BOOTSTRAP_KNOWN_WORDS = 60        # below this, level is book-count driven, not ZPD


def empty_curriculum() -> dict:
    return {"version": 1, "level": 1.5, "known_words": {}, "shelf": {},
            "level_history": [{"cycle": 0, "level": 1.5, "reason": "start"}],
            "graduated_since_advance": 0, "last_retention_cycle": 0}


def _known_set(curriculum: dict) -> set[str]:
    """Words that count toward coverage/level.  Once any word has been
    use-tested, require the `used` tier; before that, fall back to `seen`."""
    words = curriculum["known_words"]
    any_use_tested = any(info.get("use_tested") for info in words.values())
    if any_use_tested:
        return {w for w, info in words.items() if info.get("used")}
    return {w for w, info in words.items() if info.get("books", 0) >= KNOWN_AFTER_BOOKS}


def record_word_test(curriculum: dict, word: str, used: bool | None = None,
                     explained: bool | None = None, cycle: int = 0) -> None:
    """Set a word's `used` / `explained` tier from a held-out test outcome."""
    info = curriculum["known_words"].setdefault(word, {"books": 0, "first_cycle": cycle})
    if used is not None:
        info["use_tested"] = True
        info["used"] = bool(used)
        if used:
            info.setdefault("used_cycle", cycle)
    if explained is not None:
        info["explained"] = bool(explained)
        if explained:
            info.setdefault("explained_cycle", cycle)


def register_books(curriculum: dict, books: list[dict], cycle: int) -> int:
    """books: [{title, url, source, text, event_count}].  Adds unseen ones to
    the shelf, in rotation if near the current level else shelved above it."""
    known = _known_set(curriculum)
    added = 0
    for book in books:
        bid = _book_id(book["url"], book["title"])
        if bid in curriculum["shelf"]:
            continue
        difficulty = text_difficulty(book["text"], book.get("event_count", 0), known,
                                     events=book.get("events"))
        est = difficulty["estimated_level"]
        in_reach = est <= curriculum["level"] + LEVEL_STEP
        curriculum["shelf"][bid] = {
            "title": book["title"], "url": book["url"], "source": book.get("source", ""),
            "text": book.get("text", ""),
            "difficulty": difficulty, "estimated_level": est,
            "schema": _schema_signature(book.get("verbs", [])),
            "times_read": 0, "comprehension_history": [],
            "status": "in_rotation" if in_reach else "shelved_above_level",
            "shelved_at_level": None if in_reach else est,
            "first_seen_cycle": cycle, "last_read_cycle": None}
        added += 1
    # Cold start: the level must sit near real material.  If NOTHING on the shelf
    # is within a step of the level, move the level to the easiest text -- but
    # never below `floor_level` (the highest level a real advance has reached):
    # a level once earned is not surrendered just because its band's books are
    # temporarily shelved_stuck.
    near_level = [b for b in curriculum["shelf"].values()
                  if b["estimated_level"] <= curriculum["level"] + LEVEL_STEP]
    if not near_level and curriculum["shelf"]:
        easiest = min(curriculum["shelf"].values(), key=lambda b: b["estimated_level"])
        floor = curriculum.get("floor_level", 1.0)
        target = max(floor, round(max(1.0, easiest["estimated_level"]) * 2) / 2)
        if target != curriculum["level"]:
            direction = "lower" if target < curriculum["level"] else "raise"
            curriculum["level"] = target
            curriculum["level_history"].append(
                {"cycle": cycle, "level": target,
                 "reason": f"cold start: {direction} to the easiest available book"})
        for b in curriculum["shelf"].values():
            if (b["estimated_level"] <= curriculum["level"] + LEVEL_STEP
                    and b["status"] == "shelved_above_level"):
                b["status"], b["shelved_at_level"] = "in_rotation", None
    return added


CURRICULUM_SCHEMA_VERSION = 4

# fail-closed provenance: an event is Noise's own experience ONLY if it carries
# an explicit `heuristic_self` stamp.  A missing / unknown / teacher / LLM /
# caregiver provenance is never promoted to a learning signal.
HEURISTIC_SELF = "heuristic_self"
PROVENANCE_POLICY = "fail_closed_heuristic_self_v1"


def heuristic_self_only(events: list) -> list:
    return [e for e in events if isinstance(e, dict) and e.get("provenance") == HEURISTIC_SELF]


def non_self_events(events: list) -> list:
    return [e for e in events if not (isinstance(e, dict) and e.get("provenance") == HEURISTIC_SELF)]


def book_was_read(b: dict) -> bool:
    """Conservative 'Noise actually read this book' check for legacy state that
    has no explicit reading_history: a read leaves a last_read_cycle, a
    times_read count, a comprehension_history entry, or a post-reading status."""
    return bool(b.get("times_read", 0) > 0 or b.get("last_read_cycle")
                or b.get("comprehension_history")
                or b.get("status") in ("graduated", "reread", "shelved_stuck"))


def _reading_log_ids(curriculum: dict) -> "set[str] | None":
    """The canonical read-book set, if an explicit reading log exists."""
    log = curriculum.get("reading_history") or curriculum.get("reading_log")
    if not isinstance(log, (list, dict)):
        return None
    if isinstance(log, dict):
        return {str(k) for k in log}
    ids = set()
    for entry in log:
        if isinstance(entry, dict):
            bid = entry.get("book_id") or entry.get("bid") or entry.get("id")
            if bid:
                ids.add(str(bid))
        elif isinstance(entry, str):
            ids.add(entry)
    return ids or None


def migrate_reading_state(curriculum: dict, events_store: dict, extract) -> dict:
    """Idempotent, conservative migration to the evidence-isolated schema.

    v2: difficulty recomputed from heuristic events; known_words book counts
        rebuilt (no re-read inflation); aid-graduated books returned for re-eval.
    v3: ONLY books Noise can be shown to have READ contribute vocabulary evidence
        or enter the shared events store.  A book that is merely on the shelf --
        fetched but never selected -- can no longer be the reason a word counts as
        known.  Words that lose all their support are quarantined, not deleted.

    Running it again changes nothing; re-running from v2 applies only the v3 pass.
    """
    have = curriculum.get("curriculum_schema_version", 0)
    if have >= CURRICULUM_SCHEMA_VERSION:
        return {"migrated": False, "schema_version": have}

    shelf = curriculum["shelf"]
    log_ids = _reading_log_ids(curriculum)
    if log_ids is not None:
        read_ids = {bid for bid in shelf if bid in log_ids or book_was_read(shelf[bid])}
    else:
        read_ids = {bid for bid, b in shelf.items() if book_was_read(b)}
    unread_ids = set(shelf) - read_ids

    # --- 1. RE-DERIVE every read book's events with Noise's own heuristic parser
    # (v4: fail-closed provenance).  The shared store keeps ONLY events that come
    # back explicitly stamped `heuristic_self`; a book whose text cannot be
    # re-parsed to >= 3 such events is quarantined out of the store entirely.
    # By-value-unknown legacy entries are NOT rewritten to heuristic_self -- they
    # are regenerated from the source text or dropped. ---
    reparsed = difficulty_stale = 0
    provenance_quarantined_books = provenance_quarantined_events = 0
    per_book_tokens: dict[str, set] = {}
    per_book_events: dict[str, list] = {}
    for bid, b in shelf.items():
        text = b.get("text") or ""
        evs = extract(text) if text else []                 # extract() stamps heuristic_self
        self_evs = heuristic_self_only(evs)
        provenance_quarantined_events += len(evs) - len(self_evs)
        per_book_events[bid] = self_evs
        if bid in read_ids and len(self_evs) >= 3:
            events_store[bid] = self_evs
            per_book_tokens[bid] = {t for e in self_evs
                                    for t in (e.get("subject"), e.get("obj"), e.get("verb")) if t}
        else:
            if bid in read_ids and events_store.get(bid):
                provenance_quarantined_books += 1
            events_store.pop(bid, None)          # unread, or un-reparseable as heuristic_self

    # --- 2. rebuild known_words from READ books only.  Keep the existing
    # vocabulary; never mint new words here.  book_ids = distinct READ books the
    # token appears in.  A word supported by no read book is quarantined
    # (books=0, books_unverified) with its tier / provenance kept. ---
    old = curriculum.get("known_words", {})
    token_books: dict[str, list] = {}
    for bid, toks in per_book_tokens.items():
        for tok in toks:
            token_books.setdefault(tok, []).append(bid)
    fresh: dict = {}
    quarantined = book_ids_removed = words_zeroed = 0
    for tok, prev in old.items():
        legacy_ids = set(prev.get("book_ids", []))
        read_book_ids = sorted(set(token_books.get(tok, [])))
        book_ids_removed += len(legacy_ids - set(read_book_ids))
        entry = {k: prev[k] for k in ("use_tested", "used", "explained",
                                      "used_cycle", "explained_cycle", "first_cycle")
                 if k in prev}
        entry["book_ids"] = read_book_ids
        entry["books"] = len(read_book_ids)
        if not read_book_ids:
            entry["books_unverified"] = True
            quarantined += 1
            if legacy_ids:
                words_zeroed += 1
        fresh[tok] = entry
    words_before, words_after = len(old), len(fresh)
    words_retained = words_after - quarantined
    curriculum["known_words"] = fresh
    known = _known_set(curriculum)

    # --- 3. recompute difficulty AND schema from the re-derived heuristic_self
    # events (an old `schema` may have been built from teacher verbs). ---
    for bid, b in shelf.items():
        evs = per_book_events[bid]
        b["difficulty"] = text_difficulty(b.get("text") or "", len(evs), known,
                                          events=evs or None)
        b["estimated_level"] = b["difficulty"]["estimated_level"]
        b["schema"] = _schema_signature([e.get("verb", "") for e in evs]) if evs else []
        if not evs:
            b["difficulty_stale"] = True
            difficulty_stale += 1
        else:
            b.pop("difficulty_stale", None)
            reparsed += 1

    # --- 4. audit graduated books (v2 pass, still needed on a fresh v0/v1 state) ---
    re_eval = 0
    for bid, b in shelf.items():
        if b.get("status") != "graduated":
            continue
        aided = b.get("scaffolded") or len([e for e in events_store.get(bid, [])
                                            if e.get("verb")]) < 3
        if aided:
            b["status"] = "in_rotation"
            b["graduated_via_aid"] = True
            b["re_eval_pending"] = True
            b.setdefault("shelved_at_level", None)
            re_eval += 1

    curriculum["curriculum_schema_version"] = CURRICULUM_SCHEMA_VERSION
    curriculum["provenance_policy"] = PROVENANCE_POLICY
    audit = curriculum.setdefault("migration_audit", [])
    result = {"migrated": True, "from_schema": have, "to_schema": CURRICULUM_SCHEMA_VERSION,
              "provenance_policy": PROVENANCE_POLICY,
              "books_read": len(read_ids), "books_unread": len(unread_ids),
              "books_reparsed": reparsed, "books_difficulty_stale": difficulty_stale,
              "provenance_quarantined_books": provenance_quarantined_books,
              "provenance_quarantined_events": provenance_quarantined_events,
              "known_words_before": words_before, "known_words_after": words_after,
              "known_words_quarantined": quarantined,
              "unread_book_ids_removed": book_ids_removed,
              "known_words_zeroed_by_v3": words_zeroed,
              "known_words_retained": words_retained,
              "graduated_returned_for_re_eval": re_eval}
    audit.append(result)
    return result


def reevaluate_stale_parses(curriculum: dict) -> list[str]:
    """A book set aside as unparsable / shelved_stuck under an older extractor
    (`select_next_book` only ever un-shelves `shelved_above_level`).  When the
    parser version has moved on, put those books back in rotation and return
    their ids so the caller can drop their cached events for a fresh reading.
    Graduated books keep their standing; their comprehension history is kept."""
    from japanese_event_v1 import PARSER_VERSION
    reset = []
    for bid, b in curriculum["shelf"].items():
        if (b.get("status") in ("shelved_stuck", "unparsable")
                and b.get("parser_version", 0) < PARSER_VERSION):
            b["status"] = "in_rotation"
            b["shelved_at_level"] = None
            b["times_read"] = 0
            b["parser_version"] = PARSER_VERSION      # don't loop on it next cycle
            reset.append(bid)
    return reset


def reset_level_for_new_parser(curriculum: dict, cycle: int) -> dict:
    """One-off when the parser version moves: the level was inflated by the old
    extractor's over-generous scoring, so drop it back to the easiest real book
    and re-walk the corpus from there.  Graduated books stay graduated (retention
    checks re-test them); everything else re-shelves against the new level."""
    from japanese_event_v1 import PARSER_VERSION
    if curriculum.get("level_parser_version", 0) >= PARSER_VERSION:
        return {"reset": False}
    curriculum["level_parser_version"] = PARSER_VERSION
    real = [b["estimated_level"] for b in curriculum["shelf"].values()
            if not b.get("scaffolded") and b["status"] != "graduated"]
    if not real:
        return {"reset": False}
    target = round(max(1.0, min(real)) * 2) / 2
    if target >= curriculum["level"]:
        return {"reset": False}
    old = curriculum["level"]
    curriculum["level"] = target
    curriculum["floor_level"] = target           # re-walk: the old earned floor is dropped too
    curriculum["graduated_since_advance"] = 0
    curriculum["level_history"].append(
        {"cycle": cycle, "level": target,
         "reason": f"parser v{PARSER_VERSION}: re-walk from the easiest book (was {old})"})
    for b in curriculum["shelf"].values():
        if b["status"] == "graduated":
            continue
        in_reach = b["estimated_level"] <= target + LEVEL_STEP
        b["status"] = "in_rotation" if in_reach else "shelved_above_level"
        b["shelved_at_level"] = None if in_reach else b["estimated_level"]
    return {"reset": True, "from": old, "to": target}


def select_next_book(curriculum: dict) -> str | None:
    level = curriculum["level"]
    rotation = [(bid, b) for bid, b in curriculum["shelf"].items()
                if b["status"] == "in_rotation"]
    if not rotation:
        # never idle: pull the closest shelved-above book as a stretch read
        # (the caller should also request more books at this level)
        stretch = [(bid, b) for bid, b in curriculum["shelf"].items()
                   if b["status"] == "shelved_above_level"]
        if stretch:
            bid, b = min(stretch, key=lambda item: item[1]["estimated_level"])
            b["status"] = "in_rotation"
            return bid
        # otherwise give a shelved_stuck book another attempt rather than idling
        # or dropping the level -- fresh eyes (parser upgrades, grown vocabulary)
        stuck = [(bid, b) for bid, b in curriculum["shelf"].items()
                 if b["status"] == "shelved_stuck"]
        if not stuck:
            return None
        bid, b = min(stuck, key=lambda item: (item[1].get("last_read_cycle") or 0,
                                              item[1]["estimated_level"]))
        b["status"], b["times_read"] = "in_rotation", 0
        return bid

    familiar = familiar_schema(curriculum)

    def priority(item):
        bid, b = item
        cov = b["difficulty"].get("known_word_coverage", 0.5)
        # inside the ZPD is best; then prefer near the current level and least-read
        in_zpd = 0 if ZPD_KNOWN_LOW <= cov <= ZPD_KNOWN_HIGH else 1
        # among otherwise-equal books, prefer one whose schema overlaps what the
        # reader already understands (build on familiar structure first)
        schema = set(b.get("schema", []))
        familiarity = len(schema & familiar) / len(schema) if schema else 0.0
        return (in_zpd, round(abs(b["estimated_level"] - level), 2), b["times_read"],
                -round(familiarity, 2), b.get("first_seen_cycle", 0))

    return min(rotation, key=priority)[0]


def record_reading(curriculum: dict, book_id: str, events: list[dict],
                   cycle: int, comprehension: float | None = None,
                   model=None, vocab_events=None) -> dict:
    book = curriculum["shelf"].get(book_id)
    if not book:
        return {"status": "unknown_book"}
    # fail-closed: only events with an explicit `heuristic_self` stamp drive this
    # record.  Missing / unknown / teacher / LLM provenance is dropped, not
    # promoted -- a future analyser that forgets to stamp cannot leak in.
    events = heuristic_self_only(events)
    vocab_events = heuristic_self_only(vocab_events) if vocab_events is not None else events
    from japanese_event_v1 import PARSER_VERSION
    book["parser_version"] = PARSER_VERSION       # which extractor produced this reading
    scorable = [e for e in events if e.get("verb")]
    if len(scorable) < 3:
        # Not enough structure to score -- the parser (or the LLM scaffold)
        # could not turn this text into events.  Set it aside as unparsable
        # rather than rereading it six times and calling it "stuck": it does
        # not count as a comprehension failure and returns if the parser improves.
        book["times_read"] = book.get("times_read", 0) + 1
        book["last_read_cycle"] = cycle
        book["status"] = "unparsable"
        return {"status": "unparsable", "comprehension": None,
                "times_read": book["times_read"], "book_level": book["estimated_level"]}
    if comprehension is not None:
        score = comprehension
    elif model is not None:                       # reading_comprehension_v1 model
        from reading_comprehension_v1 import book_comprehension
        score = book_comprehension(events, model, _known_set(curriculum))["score"]
    else:
        score = _self_consistency(events)         # cold-start proxy
    book["times_read"] += 1
    book["last_read_cycle"] = cycle
    book["comprehension_history"].append(round(score, 3))
    if not book.get("schema"):
        book["schema"] = _schema_signature([e.get("verb", "") for e in events])

    # grow known vocabulary from this book's HEURISTIC events (subjects / objects
    # / verbs) -- `books` counts DISTINCT books a word was seen in, so a re-read
    # of the same book does not inflate it (KNOWN_AFTER_BOOKS wants two books)
    tokens = {t for e in vocab_events
              for t in (e.get("subject"), e.get("obj"), e.get("verb")) if t}
    for token in tokens:
        info = curriculum["known_words"].setdefault(
            token, {"books": 0, "first_cycle": cycle, "book_ids": []})
        if book_id not in info.setdefault("book_ids", []):
            info["book_ids"].append(book_id)
            info["books"] = len(info["book_ids"])

    history = book["comprehension_history"]
    if score >= GRADUATE_COMPREHENSION:
        book["status"] = "graduated"
        book["graduated_cycle"] = cycle
        if book.pop("re_eval_pending", None):
            book.pop("graduated_via_aid", None)      # re-confirmed on heuristic events
        else:
            curriculum["graduated_since_advance"] = curriculum.get("graduated_since_advance", 0) + 1
        outcome = "graduated"
    elif book["times_read"] >= MAX_REREADS and (
            len(history) < 3 or history[-1] <= max(history[:-1]) + 0.02):
        book["status"] = "shelved_stuck"
        book["shelved_at_level"] = book["estimated_level"]
        outcome = "shelved_no_progress"
    else:
        outcome = "reread"
    return {"status": outcome, "comprehension": round(score, 3),
            "times_read": book["times_read"], "book_level": book["estimated_level"]}


def maybe_advance_level(curriculum: dict, cycle: int) -> dict:
    at_band = [b for b in curriculum["shelf"].values()
               if abs(b["estimated_level"] - curriculum["level"]) <= LEVEL_STEP]
    graduated_band = [b for b in at_band
                      if b["status"] == "graduated" and b["comprehension_history"]]
    ungraduated_band = [b for b in at_band if b["status"] == "in_rotation"]
    dead_band = [b for b in at_band if b["status"] in ("shelved_stuck", "unparsable")]
    # advance when the band is essentially exhausted: enough graduates OR every
    # available book at this level has been graduated (a thin shelf must not trap)
    need = min(GRADUATES_TO_ADVANCE, max(1, len(at_band) - len(dead_band)))
    if not graduated_band or len(graduated_band) < need or ungraduated_band:
        return {"advanced": False,
                "reason": f"{len(graduated_band)}/{need} band books graduated, "
                          f"{len(ungraduated_band)} still in rotation"}
    recent_scores = [b["comprehension_history"][-1] for b in graduated_band[-GRADUATES_TO_ADVANCE:]]
    mean_recent = sum(recent_scores) / len(recent_scores)
    # a recent drop on current-band material pauses advancement
    in_rotation_recent = [b["comprehension_history"][-1] for b in curriculum["shelf"].values()
                          if b["status"] == "in_rotation" and b["comprehension_history"]
                          and abs(b["estimated_level"] - curriculum["level"]) <= LEVEL_STEP]
    if in_rotation_recent and sum(in_rotation_recent) / len(in_rotation_recent) < (
            ADVANCE_COMPREHENSION - STALL_COMPREHENSION_DROP):
        return {"advanced": False, "reason": "comprehension on current band dropped"}
    if mean_recent < ADVANCE_COMPREHENSION:
        return {"advanced": False, "reason": f"mean comprehension {mean_recent:.2f} below bar"}
    want_explained = explained_ratio_target(curriculum["level"])
    if want_explained > 0:
        recent = sorted(curriculum["known_words"].values(),
                        key=lambda i: i.get("used_cycle", 0), reverse=True)[:40]
        tested = [i for i in recent if i.get("used")]
        got = (sum(bool(i.get("explained")) for i in tested) / len(tested)) if tested else 0.0
        if got < want_explained:
            return {"advanced": False,
                    "reason": f"explained-tier {got:.0%} < {want_explained:.0%} target for this level"}

    new_level = round(curriculum["level"] + LEVEL_STEP, 2)
    curriculum["level"] = new_level
    curriculum["floor_level"] = new_level        # an earned level is never surrendered
    curriculum["graduated_since_advance"] = 0
    curriculum["level_history"].append({"cycle": cycle, "level": new_level,
                                        "reason": f"{len(recent_scores)} graduates, "
                                        f"mean comprehension {mean_recent:.2f}"})
    unshelved = 0
    for b in curriculum["shelf"].values():
        was = b["status"]
        if was in ("shelved_above_level", "shelved_stuck") and (
                b["shelved_at_level"] or 9) <= new_level + LEVEL_STEP:
            b["status"] = "in_rotation"
            b["shelved_at_level"] = None
            if was == "shelved_stuck":
                b["times_read"] = 0          # a fresh chance at the higher level
            unshelved += 1
    return {"advanced": True, "level": new_level, "mean_comprehension": round(mean_recent, 3),
            "unshelved": unshelved}


def retention_check_due(curriculum: dict, cycle: int) -> str | None:
    if cycle - curriculum.get("last_retention_cycle", 0) < RETENTION_INTERVAL:
        return None
    lower = [(bid, b) for bid, b in curriculum["shelf"].items()
             if b["status"] == "graduated" and b["estimated_level"] <= curriculum["level"] - LEVEL_STEP]
    if not lower:
        return None
    curriculum["last_retention_cycle"] = cycle
    return min(lower, key=lambda item: item[1].get("graduated_cycle", 0))[0]


def current_milestone(level: float) -> str:
    label = MILESTONES[0][1]
    for threshold, name in MILESTONES:
        if level + 1e-6 >= threshold:
            label = name
    return label


def reading_age(level: float) -> str:
    """Interpretable readout: level 1.5 ~ 4 years, level 9 ~ 15 years (中3)."""
    years = 4 + (level - 1.5) * 1.5
    return f"おおよそ{years:.0f}歳向けの本"


def summary(curriculum: dict) -> dict:
    by_status = Counter(b["status"] for b in curriculum["shelf"].values())
    known = _known_set(curriculum)
    words = curriculum["known_words"]
    graduated = [b for b in curriculum["shelf"].values() if b["status"] == "graduated"]
    recent = [b["comprehension_history"][-1] for b in graduated if b["comprehension_history"]][-10:]
    return {
        "level": curriculum["level"],
        "milestone": current_milestone(curriculum["level"]),
        "reading_age": reading_age(curriculum["level"]),
        "explained_ratio_target": explained_ratio_target(curriculum["level"]),
        "vocabulary_seen": sum(1 for i in words.values() if i.get("books", 0) >= KNOWN_AFTER_BOOKS),
        "vocabulary_used": sum(1 for i in words.values() if i.get("used")),
        "vocabulary_explained": sum(1 for i in words.values() if i.get("explained")),
        "known_words": len(known),
        "shelf_total": len(curriculum["shelf"]),
        "in_rotation": by_status.get("in_rotation", 0),
        "graduated": by_status.get("graduated", 0),
        "shelved_above_level": by_status.get("shelved_above_level", 0),
        "shelved_stuck": by_status.get("shelved_stuck", 0),
        "unparsable": by_status.get("unparsable", 0),
        "graduated_since_advance": curriculum.get("graduated_since_advance", 0),
        "mean_recent_comprehension": round(sum(recent) / len(recent), 3) if recent else None,
        "level_advances": len(curriculum["level_history"]) - 1,
    }
