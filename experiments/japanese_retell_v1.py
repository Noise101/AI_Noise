#!/usr/bin/env python3
"""Retelling as a comprehension signal for the developmental reading loop.

A child who has understood a story can retell it: recall the events, keep them
in order, and say who did what.  Here Noise retells from its OWN extracted
structure (japanese_event_v1) -- particle-ordered clauses, verbs conjugated
back to polite past -- and the retelling is scored by re-parsing it and
comparing the recovered (subject, verb, obj) triples and their order against
the original.

Two uses, mirroring the rest of the stack:

  * `retell(events)` gives the curriculum / caregiver loop a plain-Japanese
    artifact of "what Noise understood" from a book it just read.
  * `evaluate_retelling(stories, previous)` is the frozen, source-disjoint
    capability measurement: on held-out stories, an in-order retelling must
    survive the generate -> re-understand round trip better than a
    shuffled-order retelling of the same events (which loses the ordering
    signal and, through zero-anaphora resolution, corrupts subjects too).
    One-sided significance, credited only after two consecutive cycles --
    the same discipline as event_structure_v1 / reading_comprehension_v1.

No network, no LLM.  A free-generation retelling from a Japanese-trained
character RNN is a later step (needs the RNN trained on Japanese first); this
module is the template baseline it will be measured against.
"""

from __future__ import annotations

import hashlib
import math
import random

import japanese_event_v1 as jevent

SIGNIFICANCE_Z = 3.0
MIN_TRAIN_STORIES = 20
MIN_TEST_STORIES = 8

# dictionary form -> polite past, taken from the verbs japanese_event_v1 already
# knows (its VERB_TABLE maps surface -> dictionary; invert the polite entries,
# preferring the surface that keeps the dictionary form's leading kanji)
_POLITE_PAST: dict[str, str] = {}
for _surface, _dic in jevent.VERB_TABLE.items():
    if not _surface.endswith("ました"):
        continue
    if _dic not in _POLITE_PAST or _surface[0] == _dic[0]:
        _POLITE_PAST[_dic] = _surface
_I_ROW = {"う": "い", "く": "き", "ぐ": "ぎ", "す": "し", "つ": "ち",
          "ぬ": "に", "ぶ": "び", "む": "み", "る": "り"}
_ICHIDAN_TAIL = set("えけせてねへめれげぜでべぺいきしちにひみりぎじぢびぴ")
_ICHIDAN_KANJI = {"見る", "着る", "寝る", "出る", "居る", "得る", "似る", "煮る", "経る"}
_IRREGULAR = {"する": "しました", "くる": "きました", "来る": "来ました",
              "ある": "ありました", "いる": "いました", "行く": "行きました"}


def polite_past(verb: str) -> str:
    """Best-effort dictionary-form -> ~ました.  Heuristic for verbs outside the
    table; retelling tolerates the odd wrong conjugation."""
    if not verb:
        return verb
    if verb in _POLITE_PAST:
        return _POLITE_PAST[verb]
    if verb in _IRREGULAR:
        return _IRREGULAR[verb]
    if verb.endswith("る") and (verb in _ICHIDAN_KANJI
                                or (len(verb) >= 2 and verb[-2] in _ICHIDAN_TAIL)):
        return verb[:-1] + "ました"
    if verb[-1] in _I_ROW:
        return verb[:-1] + _I_ROW[verb[-1]] + "ました"
    return verb + "ました"


# particle order in a plain declarative clause: topic/subject, obliques, object
_ROLE_ORDER = ("は", "が", "に", "へ", "で", "と", "から", "を")


def verbalise_event(event: dict, drop_subject: bool = False) -> str:
    parts: list[str] = []
    subject = event.get("subject") or ""
    if subject and not drop_subject:
        parts.append(f"{subject}が")
    roles = event.get("roles") or {}
    for particle in _ROLE_ORDER:
        if particle in ("は", "が", "を"):
            continue
        noun = roles.get(particle)
        if noun:
            parts.append(f"{noun}{particle}")
    obj = event.get("obj") or ""
    if obj:
        parts.append(f"{obj}を")
    verb = polite_past(event.get("verb") or "")
    if not verb:
        return ""
    parts.append(verb)
    return "".join(parts) + "。"


def retell(events: list[dict], max_sentences: int = 12,
           drop_subjects: bool = False) -> str:
    out = []
    for event in events[:max_sentences]:
        clause = verbalise_event(event, drop_subject=drop_subjects)
        if clause:
            out.append(clause)
    return "".join(out)


# --- scoring -------------------------------------------------------------
def _triples(events: list[dict]) -> list[tuple[str, str, str]]:
    return [(e.get("subject", ""), e.get("verb", ""), e.get("obj", ""))
            for e in events if e.get("verb")]


def _pair_multiset(triples: list[tuple[str, str, str]]) -> "dict[tuple[str, str], int]":
    counts: dict[tuple[str, str], int] = {}
    for subject, verb, _ in triples:
        counts[(subject, verb)] = counts.get((subject, verb), 0) + 1
    return counts


def _multiset_overlap(a: dict, b: dict) -> int:
    return sum(min(count, b.get(key, 0)) for key, count in a.items())


def _order_correlation(original: list, retold: list) -> float:
    """Kendall-tau-ish: of the verbs shared between the two sequences, the
    fraction of pairs in the same relative order."""
    o_pos = {}
    for i, (_, verb, _) in enumerate(original):
        o_pos.setdefault(verb, i)
    seq = [o_pos[v] for _, v, _ in retold if v in o_pos]
    if len(seq) < 2:
        return 1.0 if seq else 0.0
    concordant = total = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            total += 1
            concordant += seq[i] <= seq[j]
    return concordant / total if total else 1.0


def score_retelling(original_events: list[dict], retold_text: str) -> dict:
    original = _triples(original_events)
    # Noise knows these nouns (it read them); give the re-parser the same
    # vocabulary rather than re-inducing boundaries from a short retelling.
    known = {t for e in original_events
             for t in (e.get("subject"), e.get("obj")) if t and len(t) >= 2}
    retold = _triples([e.__dict__ for e in jevent.extract_story(retold_text, known)])
    orig_pairs = _pair_multiset(original)
    retold_pairs = _pair_multiset(retold)
    overlap = _multiset_overlap(orig_pairs, retold_pairs)
    total_orig = sum(orig_pairs.values())
    total_retold = sum(retold_pairs.values())
    recall = overlap / total_orig if total_orig else 0.0
    precision = overlap / total_retold if total_retold else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
    order = _order_correlation(original, retold)
    fidelity = round(0.6 * f1 + 0.4 * order, 3)
    return {"recall": round(recall, 3), "precision": round(precision, 3),
            "f1": round(f1, 3), "order_correlation": round(order, 3),
            "fidelity": fidelity,
            "original_events": total_orig, "retold_events": total_retold}


# --- frozen-benchmark capability measurement ---------------------------
def _held_out(url: str) -> bool:
    return int(hashlib.sha256(f"retell:{url}".encode()).hexdigest(), 16) % 5 == 0


def evaluate_retelling(stories: list[dict], previous: dict | None = None) -> dict:
    """stories: [{url, events}].  Held-out (source-disjoint) stories are retold
    in order and, as a baseline, in a shuffled order; the capability is the
    in-order fidelity gain."""
    previous = previous or {}
    train = [s for s in stories if not _held_out(s["url"]) and len(s.get("events", [])) >= 3]
    test = [s for s in stories if _held_out(s["url"]) and len(s.get("events", [])) >= 3]
    if len(train) < MIN_TRAIN_STORIES or len(test) < MIN_TEST_STORIES:
        return {"version": 1, "status": "insufficient_stories",
                "train_stories": len(train), "test_stories": len(test),
                "fidelity": None, "fidelity_baseline": None, "beats_baseline": False,
                "significant_streak": 0,
                "learning_curve": list(previous.get("learning_curve", []))}

    gains, ordered_scores, shuffled_scores = [], [], []
    for story in test:
        events = story["events"]
        rng = random.Random(int(hashlib.sha256(story["url"].encode()).hexdigest(), 16) % (2 ** 32))
        shuffled = events[:]
        rng.shuffle(shuffled)
        ordered = score_retelling(events, retell(events))["fidelity"]
        scrambled = score_retelling(events, retell(shuffled))["fidelity"]
        ordered_scores.append(ordered)
        shuffled_scores.append(scrambled)
        gains.append(ordered - scrambled)

    n = len(gains)
    mean_gain = sum(gains) / n
    mean_ordered = sum(ordered_scores) / n
    mean_shuffled = sum(shuffled_scores) / n
    var = sum((g - mean_gain) ** 2 for g in gains) / max(1, n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    z = mean_gain / se if se > 0 else (99.0 if mean_gain > 0 else 0.0)
    p = round(0.5 * math.erfc(z / math.sqrt(2)), 6) if z > 0 else 1.0

    significant = z >= SIGNIFICANCE_Z and n >= MIN_TEST_STORIES
    prior_sig = previous.get("beats_baseline_significant", False)
    streak = previous.get("significant_streak", 0) + 1 if significant else 0

    curve = list(previous.get("learning_curve", []))
    point = {"train_stories": len(train), "fidelity": round(mean_ordered, 3),
             "fidelity_baseline": round(mean_shuffled, 3), "gain_z": round(z, 2)}
    if not curve or curve[-1]["train_stories"] != len(train):
        curve.append(point)
    curve = curve[-200:]
    tail = [c["fidelity"] for c in curve[-8:]]
    trend = "insufficient_data"
    if len(tail) >= 4:
        older = sum(tail[:len(tail) // 2]) / (len(tail) // 2)
        newer = sum(tail[len(tail) // 2:]) / (len(tail) - len(tail) // 2)
        trend = "improving" if newer > older + 0.01 else "declining" if newer < older - 0.01 else "flat"

    return {
        "version": 1, "status": "measured",
        "train_stories": len(train), "test_stories": n,
        "fidelity": round(mean_ordered, 3),
        "fidelity_baseline": round(mean_shuffled, 3),
        "fidelity_gain": round(mean_gain, 3),
        "gain_z": round(z, 2), "gain_p_one_sided": p,
        "beats_baseline_significant": significant,
        "beats_baseline": significant and (prior_sig or previous.get("significant_streak", 0) >= 1),
        "significant_streak": streak,
        "learning_curve": curve, "retelling_trend": trend,
        "limitations": ["template retelling from Noise's own events; fidelity credit "
                        "only when in-order beats shuffled-order on held-out stories, "
                        "twice.  A free-generation retelling (RNN) is a later step."],
    }


def main() -> None:
    import json
    import sys
    text = sys.stdin.read()
    events = [e.__dict__ for e in jevent.extract_story(text)]
    retold = retell(events)
    print(retold)
    print(json.dumps(score_retelling(events, retold), ensure_ascii=False))


if __name__ == "__main__":
    main()
