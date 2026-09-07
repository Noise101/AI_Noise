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
from collections import Counter

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


# --- coherence ----------------------------------------------------------
# score_retelling's f1 is a generate -> re-parse round trip: because retell()
# serialises the very triples the re-parser then recovers, f1 stays high even
# when every clause is "それが<壊れた動詞>" -- exactly the output a caregiver
# flags as 意味不明.  retelling_coherence looks at the surface instead and is
# folded into `fidelity` so the number the loop reads reflects readable
# Japanese, not round-trip stability.
_DEICTIC_SUBJECTS = {"それ", "これ", "あれ", "そこ", "ここ", "あそこ", "こう", "そう",
                     "どれ", "どこ", "どちら", "なに", "なん"}
_ADVERB_NOT_VERB = {"まもなく", "しばらく", "いったい", "やがて", "すぐ", "ずっと",
                    "きっと", "とうとう", "だんだん", "もう", "まだ", "すこし",
                    "たいへん", "どうして", "なぜ", "たぶん", "きゅうに"}
_NOMINAL_IN_VERB = ("あげく", "はず", "ため", "こと", "とき", "ところ", "人", "ひと")
_COPULA_TAIL = ("です", "ます", "である", "だった", "でした", "だろう", "でしょう")
# a body part, or an abstract event / emotion noun, as the agent of an action
# verb ("おなかが言いました", "あらそいが逃げ出しました") is a parse error -- these
# are subjects only of sensation / state / inception predicates.
_NON_AGENT_SUBJECTS = {"おなか", "はら", "のど", "むね", "せなか", "こし", "あたま",
                       "かた", "ひざ", "て", "あし", "ゆび", "め", "みみ", "はな",
                       "くち", "は", "かお", "かげ", "こえ", "なみだ", "きもち",
                       "あらそい", "けんか", "さわぎ", "よろこび", "かなしみ",
                       "いかり", "おどろき", "こえ"}
_NON_AGENTIVE_OK_VERBS = {"すく", "へる", "かわく", "いたい", "いたむ", "つく", "する",
                          "さめる", "まわる", "たつ", "でる", "とまる", "なる",
                          "ある", "いる", "おこる", "はじまる", "おわる", "つづく",
                          "あらわれる", "きえる", "みだれる"}


def _implausible_verb(verb: str) -> bool:
    """The verb slot holds something that is not a plausible predicate: a
    fragment left by mis-segmentation, a stranded particle, or an adverb.
    Kept high-precision: を/へ never sit inside a real verb, は only ever
    starts one (はなす, はこぶ), and だろう/です mark a copula, not a verb."""
    if not verb or len(verb) <= 1:
        return True
    if verb in _ADVERB_NOT_VERB:
        return True
    if "を" in verb or "へ" in verb or "は" in verb[1:]:
        return True
    if any(nom in verb for nom in _NOMINAL_IN_VERB):
        return True
    if verb.endswith(_COPULA_TAIL) or "かもしれ" in verb:
        return True
    return False


def _fragment_subject(subject: str) -> bool:
    """A dropped subject is fine; a sentence fragment or a trailing-particle
    noun phrase in the subject slot is not."""
    if not subject:
        return False
    if len(subject) > 8:
        return True
    if subject.endswith(("の", "な", "は", "が", "を", "に", "で", "と", "も", "、")):
        return True
    return subject in _ADVERB_NOT_VERB


def retelling_coherence(events: list[dict]) -> float:
    """Is the template retelling readable Japanese, independent of the
    generate -> re-parse round trip?  1.0 = nothing wrong; degrades toward 0
    for a deictic placeholder carrying every clause, verbs that are really
    mis-segmented particles or adverbs, fragment subjects, a body part or
    abstract noun as the agent of an action verb, and a "protagonist" that
    never appears in the source text."""
    scored = [e for e in events if e.get("verb")]
    if len(scored) < 2:
        return 1.0
    n = len(scored)
    subjects = [e.get("subject") or "" for e in scored]
    verbs = [e.get("verb") or "" for e in scored]
    sentences = [e.get("sentence") or "" for e in scored]

    named = [s for s in subjects if s]
    modal, modal_ct = Counter(named).most_common(1)[0] if named else ("", 0)
    modal_share = modal_ct / n
    deictic = modal in _DEICTIC_SUBJECTS and modal_share >= 0.5

    broken = sum(_implausible_verb(v) for v in verbs) / n
    frag = sum(_fragment_subject(s) for s in subjects) / n
    verb_variety = len(set(verbs)) / n
    # selectional restriction: a body part / abstract noun doing an action it
    # cannot do ("おなかが言う", "あらそいが逃げ出す")
    body_verb = sum(s in _NON_AGENT_SUBJECTS and v not in _NON_AGENTIVE_OK_VERBS
                    for s, v in zip(subjects, verbs)) / n
    # the "protagonist" the retelling names appears in NONE of the source
    # sentences -- it came from a counter word or adverb the parser mis-took for
    # a subject, not a real entity (a real protagonist survives at least once
    # even under heavy zero-anaphora)
    modal_in_source = any(modal and modal in src for src in sentences)
    stray_protagonist = (bool(modal) and modal_share >= 0.5
                         and any(sentences) and not modal_in_source)

    penalty = (0.55 if deictic else 0.0)
    penalty += 0.60 * broken
    penalty += 0.30 * frag
    penalty += 0.50 * body_verb
    penalty += 0.45 if stray_protagonist else 0.0
    penalty += 0.25 * max(0.0, 0.55 - verb_variety)      # near-total verb repetition
    # one subject mechanically opening most clauses is only a defect when the
    # clauses themselves are degraded -- otherwise it is ordinary zero-anaphora
    if not deictic and modal_share >= 0.5:
        penalty += 0.30 * modal_share * (broken + frag)
    return round(max(0.0, min(1.0, 1.0 - penalty)), 3)


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
    structural = round(0.6 * f1 + 0.4 * order, 3)
    coherence = retelling_coherence(original_events)
    # a retelling that reads as broken Japanese is not a faithful retelling,
    # however cleanly its triples survive the round trip
    fidelity = round(structural * coherence, 3)
    return {"recall": round(recall, 3), "precision": round(precision, 3),
            "f1": round(f1, 3), "order_correlation": round(order, 3),
            "structural_fidelity": structural, "coherence": coherence,
            "fidelity": fidelity,
            "original_events": total_orig, "retold_events": total_retold}


# --- frozen-benchmark capability measurement ---------------------------
EVAL_REGIME = "event_conditioned_generation_v1"
SIGNIFICANT_TRAIN_GROWTH = 1.4


def _collection(url: str) -> str:
    """Same grouping as reading_comprehension_v1._collection: a source's parts
    (an Aozora author's files dir, a wiki collection's subpages) never straddle
    train and test; a bare /wiki/Title page is its own collection."""
    base = url.split("#")[0].split("?")[0].rstrip("/")
    parts = base.split("/")
    if len(parts[3:]) >= 3:
        return "/".join(parts[:-1])
    return base


def _held_out(url: str) -> bool:
    """Hold out whole COLLECTIONS (an Aozora author's works share one) so a
    source never straddles train and test.  A standalone work is its own
    collection."""
    return int(hashlib.sha256(f"retell:{_collection(url)}".encode()).hexdigest(), 16) % 5 == 0


def _fingerprint(snapshot: list) -> str:
    return hashlib.sha256("\n".join(sorted(s["url"] for s in snapshot)).encode()).hexdigest()[:16]


def _events_seed(events: list[dict]) -> int:
    key = "|".join(f"{e.get('subject', '')}>{e.get('verb', '')}" for e in events)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 32)


def free_retell(rnn_state: dict | None, events: list[dict], length_per_event: int = 34) -> str:
    """Generate a Japanese retelling conditioned ONLY on Noise's own extracted
    event sequence.  Each event primes the character RNN with `subject が (object
    を) verb-stem`; the RNN produces the inflection and connective tissue itself.
    The gold sentences and any LLM rephrasing NEVER enter the prime -- only
    (subject, verb, object, order) does."""
    if not rnn_state or not rnn_state.get("vocab") or not events:
        return ""
    from japanese_sequence_v1 import TinyRNN, generate
    model = TinyRNN(rnn_state["vocab"], rnn_state)
    rng = random.Random(_events_seed(events))
    out = []
    for e in events[:12]:
        subj = (e.get("subject") or "")[:6]
        obj = (e.get("obj") or "")[:6]
        stem = (e.get("verb") or "")
        stem = stem[:-1] if len(stem) > 2 and stem[-1] in "るう" else stem
        prime = f"{subj}が" + (f"{obj}を" if obj else "") + stem
        out.append(generate(model, prime, length=length_per_event, rng=rng))
    return "".join(out)


def _event_repr(events: list[dict]) -> list[dict]:
    """The compressed representation the generator is conditioned on: subject,
    verb, object, order.  No sentence text."""
    return [{"subject": e.get("subject", ""), "verb": e.get("verb", ""),
             "obj": e.get("obj", ""), "roles": {}} for e in events]


def evaluate_retelling(stories: list[dict], previous: dict | None = None,
                       rnn_state: dict | None = None) -> dict:
    """Retelling capability = generating Japanese from Noise's own event
    representation that, re-parsed, preserves the story's events BETTER THAN the
    same generator given the events in a random order (the same information minus
    ordering).  The template `retell()` round-trip is a diagnostic only.

    The held-out test set is a SNAPSHOT frozen the first time it is large enough;
    new stories only grow training.
    """
    previous = previous or {}
    snap = previous.get("test_snapshot") if previous.get("eval_regime") == EVAL_REGIME else None
    migrated = bool(previous.get("snapshot_migrated"))   # sticky once migrated

    if snap:
        test = [{"url": s["url"], "events": s["events"]} for s in snap]
    else:
        train0 = [s for s in stories if not _held_out(s["url"]) and len(s.get("events", [])) >= 3]
        cand = [s for s in stories if _held_out(s["url"]) and len(s.get("events", [])) >= 3]
        if len(train0) < MIN_TRAIN_STORIES or len(cand) < MIN_TEST_STORIES:
            return {"version": 3, "status": "insufficient_stories", "eval_regime": EVAL_REGIME,
                    "train_stories": len(train0), "test_stories": len(cand),
                    "roundtrip_fidelity": None, "generation_gain": None, "beats_baseline": False,
                    "significant_streak": 0,
                    "learning_curve": list(previous.get("learning_curve", []))}
        test = [{"url": s["url"], "events": s["events"]} for s in cand]
        snap = test
        migrated = migrated or bool(previous)

    held_cols = {_collection(s["url"]) for s in snap}
    train = [s for s in stories if _collection(s["url"]) not in held_cols
             and len(s.get("events", [])) >= 3]
    n = len(test)

    # round-trip diagnostic (NOT a capability)
    diag, tmpl_base = [], []
    for s in test:
        ev = s["events"]
        rng = random.Random(_events_seed(ev))
        shuf = ev[:]
        rng.shuffle(shuf)
        diag.append(score_retelling(ev, retell(ev))["fidelity"])
        tmpl_base.append(score_retelling(ev, retell(shuf))["fidelity"])
    mean_ordered = sum(diag) / n
    mean_tmpl_shuffled = sum(tmpl_base) / n

    # the capability: the generator conditioned on the ORDERED event repr vs the
    # SAME generator conditioned on a shuffled repr (identical information content)
    gen_gain = z = None
    gen_scores, gen_base = [], []
    if rnn_state and rnn_state.get("vocab"):
        for s in test:
            repr_ = _event_repr(s["events"])
            rng = random.Random(_events_seed(s["events"]))
            shuffled_repr = repr_[:]
            rng.shuffle(shuffled_repr)
            ordered_text = free_retell(rnn_state, repr_)
            shuffled_text = free_retell(rnn_state, shuffled_repr)
            gen_scores.append(score_retelling(s["events"], ordered_text)["fidelity"] if ordered_text else 0.0)
            gen_base.append(score_retelling(s["events"], shuffled_text)["fidelity"] if shuffled_text else 0.0)
        gains = [a - b for a, b in zip(gen_scores, gen_base)]
        gen_gain = sum(gains) / n
        var = sum((g - gen_gain) ** 2 for g in gains) / max(1, n - 1)
        se = math.sqrt(var / n) if var > 0 else 0.0
        z = gen_gain / se if se > 0 else (99.0 if gen_gain > 0 else 0.0)

    if z is None:
        status, significant = "no_generation_model", False
    else:
        status, significant = "measured", (z >= SIGNIFICANCE_Z and n >= MIN_TEST_STORIES)
    last_sig_train = previous.get("last_significant_train", 0)
    grew = len(train) >= last_sig_train * SIGNIFICANT_TRAIN_GROWTH
    streak = ((previous.get("significant_streak", 0) + 1) if (significant and grew)
              else previous.get("significant_streak", 0) if significant else 0)
    beats = significant and streak >= 2

    curve = list(previous.get("learning_curve", []))
    point = {"train_stories": len(train), "roundtrip_fidelity": round(mean_ordered, 3),
             "generation_gain": None if gen_gain is None else round(gen_gain, 3),
             "gain_z": None if z is None else round(z, 2)}
    if not curve or curve[-1]["train_stories"] != len(train):
        curve.append(point)
    curve = curve[-200:]
    tail = [c.get("generation_gain") or 0.0 for c in curve[-8:]]
    trend = "insufficient_data"
    if len(tail) >= 4:
        older = sum(tail[:len(tail) // 2]) / (len(tail) // 2)
        newer = sum(tail[len(tail) // 2:]) / (len(tail) - len(tail) // 2)
        trend = "improving" if newer > older + 0.01 else "declining" if newer < older - 0.01 else "flat"

    return {
        "version": 3, "status": status, "eval_regime": EVAL_REGIME,
        "snapshot_migrated": migrated,
        "train_stories": len(train), "test_stories": n,
        "snapshot_stories": len(snap), "snapshot_fingerprint": _fingerprint(snap),
        "test_snapshot": snap,
        "roundtrip_fidelity": round(mean_ordered, 3),          # diagnostic, not capability
        "roundtrip_template_shuffled": round(mean_tmpl_shuffled, 3),
        "generation_gain": None if gen_gain is None else round(gen_gain, 3),
        "gain_z": None if z is None else round(z, 2),
        "beats_baseline_significant": significant,
        "beats_baseline": beats,
        "significant_streak": streak,
        "last_significant_train": len(train) if significant else last_sig_train,
        "learning_curve": curve, "retelling_trend": trend,
        "limitations": ["capability = the event-conditioned generator beating the "
                        "SAME generator on a shuffled event representation, on the "
                        "FROZEN snapshot, at two different training sizes.  The "
                        "template round-trip is a diagnostic and earns nothing."],
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
