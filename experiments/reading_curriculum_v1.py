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
SUBORDINATE = re.compile(r"(ので|けれど|けれども|ながら|たり|ば|たら|なら|"
                         r"ても|でも|し、|が、|のに|ように|ため|という)")

GRADUATE_COMPREHENSION = 0.75      # a book is "understood" at/above this
ADVANCE_COMPREHENSION = 0.80       # mean over recent graduates to raise the level
STALL_COMPREHENSION_DROP = 0.15    # recent comprehension this far below -> pause advancement
MAX_REREADS = 6                    # rereads before a stuck book is shelved
GRADUATES_TO_ADVANCE = 4           # graduated books at the band before advancing
LEVEL_STEP = 0.5
KNOWN_AFTER_BOOKS = 2              # a word is "known" after this many distinct books
ZPD_KNOWN_LOW, ZPD_KNOWN_HIGH = 0.88, 0.98
RETENTION_INTERVAL = 25           # cycles between retention re-tests

# reference calibration points (estimated_level for known text kinds)
#   ~1.5  first-reader picture book (very short sentences, almost no kanji)
#   ~2.2  folktale retelling in 敬体 (楠山正雄 style)
#   ~3.5  translated fairy tale / longer folktale
#   ~4.3  literary children's prose (新美南吉 ごんぎつね)
#   ~5.0  宮沢賢治


def _book_id(url: str, title: str) -> str:
    return hashlib.sha256(f"{url}|{title}".encode()).hexdigest()[:16]


def text_difficulty(text: str, event_count: int, known_words: set[str]) -> dict:
    words = WORD.findall(text)
    sentences = [s for s in SENT.split(text) if len(s) > 3]
    n_sent = max(1, len(sentences))
    total_chars = sum(len(s) for s in sentences)
    kanji = len(KANJI.findall(text))
    vocab = set(words)
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
    features["estimated_level"] = round(max(1.0, min(6.0, level)), 2)
    return features


def _self_consistency(events: list[dict]) -> float:
    """Placeholder comprehension proxy until reading_comprehension_v1 exists:
    a story is 'coherent' if one protagonist carries most events and there is a
    beginning-middle-end spread of distinct verbs."""
    if len(events) < 3:
        return 0.0
    subjects = Counter(e.get("subject", "") for e in events)
    protagonist_share = subjects.most_common(1)[0][1] / len(events)
    distinct_verbs = len({e.get("verb", "") for e in events})
    verb_spread = min(1.0, distinct_verbs / max(3, len(events) * 0.5))
    mean_conf = sum(e.get("confidence", 0.0) for e in events) / len(events)
    return round(0.4 * protagonist_share + 0.3 * verb_spread + 0.3 * mean_conf, 3)


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
        difficulty = text_difficulty(book["text"], book.get("event_count", 0), known)
        est = difficulty["estimated_level"]
        in_reach = est <= curriculum["level"] + LEVEL_STEP
        curriculum["shelf"][bid] = {
            "title": book["title"], "url": book["url"], "source": book.get("source", ""),
            "difficulty": difficulty, "estimated_level": est,
            "times_read": 0, "comprehension_history": [],
            "status": "in_rotation" if in_reach else "shelved_above_level",
            "shelved_at_level": None if in_reach else est,
            "first_seen_cycle": cycle, "last_read_cycle": None}
        added += 1
    # Cold start: there must always be something to read.  If nothing is in
    # rotation, drop the level to the easiest available book and unshelve it.
    if not any(b["status"] == "in_rotation" for b in curriculum["shelf"].values()):
        easiest = min(curriculum["shelf"].values(),
                      key=lambda b: b["estimated_level"], default=None)
        if easiest:
            target = round(max(1.0, easiest["estimated_level"]) * 2) / 2
            if target < curriculum["level"]:
                curriculum["level"] = target
                curriculum["level_history"].append(
                    {"cycle": cycle, "level": target, "reason": "cold start: no book in reach"})
            for b in curriculum["shelf"].values():
                if b["estimated_level"] <= curriculum["level"] + LEVEL_STEP:
                    b["status"], b["shelved_at_level"] = "in_rotation", None
    return added


def select_next_book(curriculum: dict) -> str | None:
    level = curriculum["level"]
    rotation = [(bid, b) for bid, b in curriculum["shelf"].items()
                if b["status"] == "in_rotation"]
    if not rotation:
        # never idle: pull the closest not-yet-graduated shelved book as a
        # stretch read (the caller should also request more books at this level)
        stretch = [(bid, b) for bid, b in curriculum["shelf"].items()
                   if b["status"] == "shelved_above_level"]
        # a stuck book only becomes readable again after a level rise
        if not stretch:
            return None
        bid, b = min(stretch, key=lambda item: item[1]["estimated_level"])
        b["status"] = "in_rotation"
        return bid

    def priority(item):
        bid, b = item
        cov = b["difficulty"].get("known_word_coverage", 0.5)
        # inside the ZPD is best; then prefer near the current level and least-read
        in_zpd = 0 if ZPD_KNOWN_LOW <= cov <= ZPD_KNOWN_HIGH else 1
        return (in_zpd, abs(b["estimated_level"] - level), b["times_read"],
                b.get("first_seen_cycle", 0))

    return min(rotation, key=priority)[0]


def record_reading(curriculum: dict, book_id: str, events: list[dict],
                   cycle: int, comprehension: float | None = None) -> dict:
    book = curriculum["shelf"].get(book_id)
    if not book:
        return {"status": "unknown_book"}
    score = comprehension if comprehension is not None else _self_consistency(events)
    book["times_read"] += 1
    book["last_read_cycle"] = cycle
    book["comprehension_history"].append(round(score, 3))

    # grow known vocabulary from this book's events (subjects/objects/verbs)
    tokens = {t for e in events for t in (e.get("subject"), e.get("obj"), e.get("verb")) if t}
    for token in tokens:
        info = curriculum["known_words"].setdefault(token, {"books": 0, "first_cycle": cycle})
        info["books"] = info.get("books", 0) + 1

    history = book["comprehension_history"]
    if score >= GRADUATE_COMPREHENSION:
        book["status"] = "graduated"
        book["graduated_cycle"] = cycle
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
    stuck_band = [b for b in at_band if b["status"] == "shelved_stuck"]
    # advance when the band is essentially exhausted: enough graduates OR every
    # available book at this level has been graduated (a thin shelf must not trap)
    need = min(GRADUATES_TO_ADVANCE, max(1, len(at_band) - len(stuck_band)))
    if (len(graduated_band) < need or ungraduated_band) and graduated_band:
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

    new_level = round(curriculum["level"] + LEVEL_STEP, 2)
    curriculum["level"] = new_level
    curriculum["graduated_since_advance"] = 0
    curriculum["level_history"].append({"cycle": cycle, "level": new_level,
                                        "reason": f"{len(recent_scores)} graduates, "
                                        f"mean comprehension {mean_recent:.2f}"})
    unshelved = 0
    for b in curriculum["shelf"].values():
        if b["status"] in ("shelved_above_level", "shelved_stuck") and (
                b["shelved_at_level"] or 9) <= new_level + LEVEL_STEP:
            b["status"] = "in_rotation"
            b["shelved_at_level"] = None
            b["times_read"] = 0 if b.get("status") == "shelved_stuck" else b["times_read"]
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


def reading_age(level: float) -> str:
    """Interpretable readout, calibrated on graded-reader difficulty (rough)."""
    years = 2 + (level - 1.0) * 1.6
    return f"おおよそ{years:.0f}歳向けの本"


def summary(curriculum: dict) -> dict:
    by_status = Counter(b["status"] for b in curriculum["shelf"].values())
    known = _known_set(curriculum)
    words = curriculum["known_words"]
    graduated = [b for b in curriculum["shelf"].values() if b["status"] == "graduated"]
    recent = [b["comprehension_history"][-1] for b in graduated if b["comprehension_history"]][-10:]
    return {
        "level": curriculum["level"],
        "reading_age": reading_age(curriculum["level"]),
        "vocabulary_seen": sum(1 for i in words.values() if i.get("books", 0) >= KNOWN_AFTER_BOOKS),
        "vocabulary_used": sum(1 for i in words.values() if i.get("used")),
        "vocabulary_explained": sum(1 for i in words.values() if i.get("explained")),
        "known_words": len(known),
        "shelf_total": len(curriculum["shelf"]),
        "in_rotation": by_status.get("in_rotation", 0),
        "graduated": by_status.get("graduated", 0),
        "shelved_above_level": by_status.get("shelved_above_level", 0),
        "shelved_stuck": by_status.get("shelved_stuck", 0),
        "graduated_since_advance": curriculum.get("graduated_since_advance", 0),
        "mean_recent_comprehension": round(sum(recent) / len(recent), 3) if recent else None,
        "level_advances": len(curriculum["level_history"]) - 1,
    }
