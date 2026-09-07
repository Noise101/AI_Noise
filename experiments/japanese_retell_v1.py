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
import json
import math
import random
import time
from collections import Counter, defaultdict

import japanese_benchmark_v1 as jb
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
# v2: the capability is NARRATIVE ORDER RECOVERY.  Given a story's events with
# their order withheld, does Noise's character RNN -- used as a likelihood model,
# not a per-event text generator -- prefer the gold ordering of each pair of
# events over the reversed one, MORE OFTEN than a simple verb-position baseline
# built from the same training corpus?  This removes the structural shortcut in
# v1, where an outer loop generated each event and concatenated them in input
# order (so "ordered" input always beat "shuffled" input regardless of the RNN).
EVAL_REGIME = "narrative_order_recovery_v2"   # v2: tiered selection + one-shot final
SCORING_VERSION = 2
BASELINE_DEFINITION = "verb_position_from_training_corpus"
BENCH_SALT = "retell:nor:v2"
SIGNIFICANT_TRAIN_GROWTH = 1.4
RETELL_REEVAL_MIN_STEP_GROWTH = 1.15   # RNN steps must grow this much (or ...
RETELL_REEVAL_MIN_STEP_ABS = 50_000    # ... this many absolute steps, whichever
                                       # is larger -- a fresh RNN is not re-scored
                                       # every 700 steps), or ...
RETELL_REEVAL_MIN_SECONDS = 6 * 3600   # ... this long must pass, or new training
MAX_ORDER_PAIRS = 15                   # event pairs scored per held-out story


_collection = jb.collection
_canonical_events = jb.canonical_events
_fingerprint = jb.fingerprint


def _held_out(url: str) -> bool:
    """Legacy per-collection ~1/5 split -- the tiered benchmark uses jb.Tiers now;
    kept for callers / tests that still reason about a single held-out set."""
    return int(hashlib.sha256(f"retell:{jb.collection(url)}".encode()).hexdigest(), 16) % 5 == 0


def forbidden_training_collections(previous: dict | None, stories: list | None = None) -> set:
    """Collections that must stay OUT of the RNN's training text for the
    retelling benchmark: every non-train tier plus every frozen selection /
    final / reserve collection."""
    previous = previous or {}
    prev = previous if previous.get("eval_regime") == EVAL_REGIME else {}
    return set(jb.Tiers(stories or [], BENCH_SALT, prev).forbidden_train_collections)


def _events_seed(events: list[dict]) -> int:
    key = "|".join(f"{e.get('subject', '')}>{e.get('verb', '')}" for e in events)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 32)


def _serialize_event(e: dict) -> str:
    s, v, o = e.get("subject", ""), e.get("verb", ""), e.get("obj", "")
    return (f"{s}{'が' if s else ''}{o}{'を' if o else ''}{polite_past(v)}。") if v else ""


def _event_repr(events: list[dict]) -> list[dict]:
    """The compressed representation the generator / scorer is conditioned on:
    subject, verb, object.  No sentence text, no order beyond list position."""
    return [{"subject": e.get("subject", ""), "verb": e.get("verb", ""),
             "obj": e.get("obj", ""), "roles": {}} for e in events]


def free_retell(rnn_state: dict | None, events: list[dict], total_length: int | None = None) -> str:
    """A single free-generation retelling for DISPLAY ONLY (status-ja).

    The RNN is primed once with a PERMUTATION-INVARIANT cue built from the whole
    event set (entities and verbs, sorted, particles only) and generates one
    continuous passage.  It is NOT primed event-by-event and the output is NOT
    concatenated in input order -- that shortcut is what the v1 capability metric
    accidentally rewarded.  This function earns no capability credit; the metric
    is `narrative order recovery` below.
    """
    if not rnn_state or not rnn_state.get("vocab") or not events:
        return ""
    from japanese_sequence_v1 import TinyRNN, generate
    model = TinyRNN(rnn_state["vocab"], rnn_state)
    triples = sorted({(e.get("subject") or "", e.get("verb") or "", e.get("obj") or "")
                      for e in events if e.get("verb")})
    cue = "、".join(t for s, v, o in triples for t in ((s, o, v) if o else (s, v)) if t)[:80]
    # seed from the permutation-invariant triple set, not event order -- the
    # whole point is that input order does not touch the output
    seed = int(hashlib.sha256("|".join(f"{s}>{v}>{o}" for s, v, o in triples).encode()).hexdigest(), 16) % (2 ** 32)
    length = total_length or max(60, 26 * min(len(triples), 12))
    return generate(model, cue or "むかし", length=length, rng=random.Random(seed))


# --- narrative order recovery ----------------------------------------------
def _position_model(train_stories: list[dict]) -> dict:
    """verb -> mean relative story position, learned from the TRAINING stories.
    The simple-method baseline: for a pair of events, predict the one whose verb
    is on-average earlier goes first.  Same information the RNN scorer gets."""
    pos: "dict[str, list]" = defaultdict(list)
    for s in train_stories:
        verbs = [e.get("verb") for e in s["events"] if e.get("verb")]
        for i, v in enumerate(verbs):
            pos[v].append(i / max(1, len(verbs) - 1))
    return {v: sum(p) / len(p) for v, p in pos.items() if p}


def _order_pairs(k: int, seed: int) -> list[tuple[int, int]]:
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    if len(pairs) <= MAX_ORDER_PAIRS:
        return pairs
    rng = random.Random(seed)
    return sorted(rng.sample(pairs, MAX_ORDER_PAIRS))


def _order_recovery(events: list[dict], scorer, mean_pos: dict) -> "tuple[float, float, int] | None":
    """For each sampled pair (i<j) of events (i truly precedes j), check whether
    the RNN likelihood-scorer prefers 'i then j' to 'j then i', and whether the
    verb-position baseline does.  Returns (rnn_accuracy, baseline_accuracy,
    n_pairs) or None when the story has too few scorable events."""
    ev = [e for e in events if e.get("verb")]
    texts = [_serialize_event(e) for e in ev]
    idx = [i for i, t in enumerate(texts) if t]
    if len(idx) < 3:
        return None
    pairs = _order_pairs(len(idx), _events_seed(ev))
    rnn_hits = base_hits = 0.0
    for a, b in pairs:
        ia, ib = idx[a], idx[b]                       # gold order: ia precedes ib
        ta, tb = texts[ia], texts[ib]
        sab, sba = scorer(ta + tb), scorer(tb + ta)   # lower bits = preferred
        rnn_hits += 1.0 if sab < sba else 0.5 if sab == sba else 0.0
        pa = mean_pos.get(ev[ia].get("verb"), 0.5)
        pb = mean_pos.get(ev[ib].get("verb"), 0.5)
        # NEVER break a tie with the index -- the snapshot stores events in gold
        # order, so `ia < ib` would leak the answer to the baseline
        base_hits += 1.0 if pa < pb else 0.5 if pa == pb else 0.0
    m = len(pairs)
    return rnn_hits / m, base_hits / m, m


def _rnn_scorer(rnn_state: dict):
    from japanese_sequence_v1 import TinyRNN
    model = TinyRNN(rnn_state["vocab"], rnn_state)

    def score(text: str) -> float:
        bpc, n = model.bits_per_char(text)
        return bpc * max(n, 1)               # total bits; lower = more likely
    return score


def _due_for_reeval(previous: dict, rnn_state: dict | None, train_n: int) -> bool:
    """Recompute only when something that feeds the measurement actually moved:
    a different RNN fingerprint AND (enough new steps, enough elapsed time, or
    new training stories)."""
    cur_fp = (rnn_state or {}).get("model_fingerprint")
    if cur_fp != previous.get("rnn_fingerprint"):
        prev_steps = previous.get("rnn_steps_at_eval", 0)
        cur_steps = (rnn_state or {}).get("steps_trained", 0)
        if train_n != previous.get("train_stories"):
            return True
        step_gate = max(prev_steps * (RETELL_REEVAL_MIN_STEP_GROWTH - 1.0), RETELL_REEVAL_MIN_STEP_ABS)
        if prev_steps and cur_steps - prev_steps >= step_gate:
            return True
        if time.time() - previous.get("evaluated_at", 0) >= RETELL_REEVAL_MIN_SECONDS:
            return True
        if not prev_steps:              # first time we ever see a model
            return True
        return False
    return False


def _measure_order_recovery(test_stories: list[dict], scorer, mean_pos: dict) -> dict | None:
    per, r_accs, b_accs = [], [], []
    for s in test_stories:
        got = _order_recovery(s["events"], scorer, mean_pos)
        if got is None:
            continue
        r, b, _m = got
        r_accs.append(r)
        b_accs.append(b)
        per.append(r - b)
    if len(per) < MIN_TEST_STORIES:
        return None
    n = len(per)
    gain = sum(per) / n
    var = sum((g - gain) ** 2 for g in per) / max(1, n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    z = gain / se if se > 0 else (99.0 if gain > 0 else 0.0)
    return {"n": n, "order_gain": round(gain, 4),
            "rnn_pairwise_accuracy": round(sum(r_accs) / n, 3),
            "position_baseline_accuracy": round(sum(b_accs) / n, 3),
            "gain_z": round(z, 2), "significant": z >= SIGNIFICANCE_Z}


def _roundtrip_diag(test_stories: list[dict]) -> tuple[float, float]:
    diag, base = [], []
    for s in test_stories:
        ev = s["events"]
        rng = random.Random(_events_seed(ev))
        shuf = ev[:]
        rng.shuffle(shuf)
        diag.append(score_retelling(ev, retell(ev))["fidelity"])
        base.append(score_retelling(ev, retell(shuf))["fidelity"])
    n = max(1, len(test_stories))
    return sum(diag) / n, sum(base) / n


def evaluate_retelling(stories: list[dict], previous: dict | None = None,
                       rnn_state: dict | None = None) -> dict:
    """Tiered NARRATIVE ORDER RECOVERY benchmark.

    SELECTION (frozen once, measured when the RNN meaningfully changed):
    diagnostic learning curve + the milestone that lets a FINAL open.  FINAL
    (collection-disjoint from train AND selection, opened at most
    FINAL_QUERY_BUDGET times on never-opened snapshots, model / regime / scoring
    / baseline / threshold frozen from the selection side first): capability =
    selection-significant AND an unopened final that also clears the threshold.
    free_retell is display only; the template round trip is a diagnostic.
    """
    previous = previous or {}
    regime_ok = previous.get("eval_regime") == EVAL_REGIME
    prev = previous if regime_ok else {}
    from japanese_event_v1 import PARSER_VERSION

    has_model = bool(rnn_state and rnn_state.get("vocab"))
    cur_fp = rnn_state.get("model_fingerprint") if has_model else None
    cur_steps = rnn_state.get("steps_trained", 0) if has_model else 0
    cur_train_regime = rnn_state.get("training_regime") if has_model else None

    tiers = jb.Tiers(stories, BENCH_SALT, prev)
    train = tiers.train_stories
    base = {"version": 5, "eval_regime": EVAL_REGIME,
            "regime_reset_from": previous.get("eval_regime") if (previous and not regime_ok) else None,
            "snapshot_migrated": tiers.selection_migrated or (bool(previous) and not regime_ok),
            **tiers.report_fields(),
            "selection_snapshot": tiers.selection_snapshot,
            "reserve_snapshot": tiers.reserve_snapshot,
            "rnn_fingerprint": cur_fp, "rnn_training_regime": cur_train_regime,
            "rnn_steps_now": cur_steps,
            "learning_curve": list(prev.get("learning_curve", []))}

    if not tiers.selection_frozen or len(train) < MIN_TRAIN_STORIES:
        return {**base, "status": "insufficient_stories", "beats_baseline": False,
                "order_gain": None, "roundtrip_fidelity": None, "recomputed": True,
                "train_stories": len(train)}

    mean_pos = _position_model([{"events": s["events"]} for s in train])

    # ---- SELECTION (repeatable, NOT capability) ----
    # carry the (minutes-long) pairwise pass forward unless the snapshot changed,
    # the training size changed, or the RNN both moved fingerprint AND crossed a
    # re-eval threshold (>=15% new steps / 6h / new training stories).
    sel_prev = prev.get("selection")
    sel_key = [cur_fp, tiers.selection_fingerprint, len(train)]
    reusable = (sel_prev is not None
                and prev.get("selection_fingerprint_at_eval") == tiers.selection_fingerprint
                and prev.get("selection_train_stories") == len(train)
                and not _due_for_reeval(
                    {"rnn_fingerprint": prev.get("selection_rnn_fingerprint"),
                     "rnn_steps_at_eval": prev.get("selection_rnn_steps_at_eval", 0),
                     "evaluated_at": prev.get("selection_evaluated_at", 0),
                     "train_stories": prev.get("selection_train_stories")},
                    rnn_state, len(train)))
    if reusable:
        sel = sel_prev
        sel_recomputed = False
        sel_steps_at_eval = prev.get("selection_rnn_steps_at_eval", cur_steps)
        sel_evaluated_at = prev.get("selection_evaluated_at", time.time())
        rt_fid = prev.get("roundtrip_fidelity")
        rt_shuf = prev.get("roundtrip_template_shuffled")
    else:
        sel_recomputed = True
        sel_steps_at_eval = cur_steps
        sel_evaluated_at = time.time()
        rt_fid, rt_shuf = _roundtrip_diag(tiers.selection_snapshot)
        sel = (_measure_order_recovery(tiers.selection_snapshot, _rnn_scorer(rnn_state), mean_pos)
               if has_model else None)

    sel_significant = bool(sel and sel["significant"])
    sel_measurements = prev.get("selection_measurements", 0) + (1 if sel_recomputed else 0)
    last_sig_train = prev.get("selection_last_significant_train", 0)
    grew = len(train) >= max(1, last_sig_train) * SIGNIFICANT_TRAIN_GROWTH
    sel_sig_streak = ((prev.get("selection_significant_streak", 0) + 1)
                      if (sel_significant and grew and sel_recomputed)
                      else prev.get("selection_significant_streak", 0) if sel_significant else 0)

    # ---- FINAL (one-shot capability gate) ----
    model_fp = "|".join(str(x) for x in (cur_fp, PARSER_VERSION, SCORING_VERSION, EVAL_REGIME))
    final_history = list(prev.get("final_history", []))
    prev_standing = final_history[-1] if final_history else None
    standing_stale_or_failed = bool(prev_standing and (
        prev_standing["preconditions"].get("model_fingerprint") != model_fp
        or not (prev_standing["result"] or {}).get("significant")))
    can_open = (has_model and sel_significant and sel_sig_streak >= 2 and tiers.disjoint
                and len(final_history) < jb.FINAL_QUERY_BUDGET
                and (not final_history or standing_stale_or_failed))
    nxt = tiers.next_unopened_final() if can_open else None
    if nxt:
        pre = {"model_fingerprint": model_fp, "rnn_fingerprint": cur_fp,
               "rnn_steps_at_open": cur_steps, "rnn_training_regime": cur_train_regime,
               "parser_version": PARSER_VERSION, "eval_regime": EVAL_REGIME,
               "scoring_version": SCORING_VERSION, "baseline_definition": BASELINE_DEFINITION,
               "significance_z": SIGNIFICANCE_Z, "selection_result": sel,
               "selection_fingerprint": tiers.selection_fingerprint,
               "final_snapshot_fingerprint": nxt["fingerprint"],
               "final_snapshot_urls": nxt["urls"],
               "final_query_index": len(final_history) + 1, "evaluated_at": time.time()}
        fin = _measure_order_recovery(nxt["snapshot"], _rnn_scorer(rnn_state), mean_pos)
        final_history = final_history + [{"tier": nxt["tier"], "fingerprint": nxt["fingerprint"],
                                          "result": fin, "preconditions": pre}]
        tiers.record_final(nxt)

    standing = final_history[-1] if final_history else None
    final_stale = bool(standing and standing["preconditions"].get("model_fingerprint") != model_fp)
    beats = bool(standing and standing["result"] and standing["result"]["significant"]
                 and sel_significant and not final_stale and tiers.disjoint)
    if standing:
        final_status = "opened" if not final_stale else "stale_needs_fresh_final"
    elif not has_model:
        final_status = "no_generation_model"
    elif tiers.next_unopened_final():
        final_status = "unopened"
    else:
        final_status = "insufficient_final_stories"

    status = "measured" if has_model else "no_generation_model"
    curve = list(prev.get("learning_curve", []))
    point = {"train_stories": len(train), "eval_regime": EVAL_REGIME,
             "selection_measurement": sel_measurements, "rnn_steps": cur_steps,
             "roundtrip_fidelity": round(rt_fid, 3) if rt_fid is not None else None,
             "order_gain": (sel or {}).get("order_gain"),
             "gain_z": (sel or {}).get("gain_z")}
    if not curve or curve[-1].get("train_stories") != len(train) or curve[-1].get("rnn_steps") != cur_steps:
        curve.append(point)
    curve = curve[-200:]
    tail = [c.get("order_gain") or 0.0 for c in curve[-8:]]
    trend = "insufficient_data"
    if len(tail) >= 4:
        older = sum(tail[:len(tail) // 2]) / (len(tail) // 2)
        newer = sum(tail[len(tail) // 2:]) / (len(tail) - len(tail) // 2)
        trend = "improving" if newer > older + 0.005 else "declining" if newer < older - 0.005 else "flat"

    return {
        **base, "status": status, "recomputed": sel_recomputed,
        "train_stories": len(train),
        "model_fingerprint": model_fp,
        "rnn_model_changed": cur_fp != prev.get("rnn_fingerprint"),
        # selection (diagnostic)
        "selection": sel,
        "selection_key": sel_key,
        "selection_fingerprint_at_eval": tiers.selection_fingerprint,
        "selection_measurements": sel_measurements,
        "selection_significant_streak": sel_sig_streak,
        "selection_last_significant_train": len(train) if sel_significant else last_sig_train,
        "selection_rnn_fingerprint": cur_fp,
        "selection_rnn_steps_at_eval": sel_steps_at_eval,
        "selection_evaluated_at": sel_evaluated_at,
        "selection_train_stories": len(train),
        "rnn_steps_at_eval": sel_steps_at_eval,
        "next_reeval": _next_reeval_hint({"rnn_steps_at_eval": sel_steps_at_eval}, cur_steps),
        "roundtrip_fidelity": round(rt_fid, 3) if rt_fid is not None else None,
        "roundtrip_template_shuffled": round(rt_shuf, 3) if rt_shuf is not None else None,
        # compat fields for status renderers / tests
        "order_gain": (sel or {}).get("order_gain"),
        "rnn_pairwise_accuracy": (sel or {}).get("rnn_pairwise_accuracy"),
        "position_baseline_accuracy": (sel or {}).get("position_baseline_accuracy"),
        "gain_z": (sel or {}).get("gain_z"),
        "beats_baseline_significant": sel_significant,
        "test_stories": (sel or {}).get("n"),
        "snapshot_fingerprint": tiers.selection_fingerprint,
        "snapshot_stories": len(tiers.selection_snapshot),
        "significant_streak": sel_sig_streak,
        # final (one-shot capability)
        "final_history": final_history,
        "final_opened_count": len(final_history),
        "final_query_budget": jb.FINAL_QUERY_BUDGET,
        "final_status": final_status,
        "final_result": standing["result"] if standing else None,
        "final_stale_for_current_model": final_stale,
        "final_preconditions": standing["preconditions"] if standing else None,
        "capability_confirmed": beats,
        "beats_baseline": beats,
        "capability_pending_reason": (None if beats else
            "final not yet opened" if final_status in ("unopened", "no_generation_model", "insufficient_final_stories") else
            "selection not significant" if not sel_significant else
            "final below threshold" if standing and not standing["result"]["significant"] else
            "final stale (model changed since it was opened)" if final_stale else "unknown"),
        "learning_curve": curve, "retelling_trend": trend,
        "limitations": ["capability = RNN-as-likelihood-model recovering gold event ORDER "
                        "(pairwise, symmetric) better than a verb-position baseline, on an "
                        "UNOPENED collection-disjoint FINAL, with model/regime/threshold "
                        "pre-registered from the SELECTION side. free_retell is display only; "
                        "an untrained RNN scores ~0.5 and cannot pass."],
    }


def _next_reeval_hint(previous: dict, cur_steps: int) -> str:
    at = max(0, previous.get("rnn_steps_at_eval", cur_steps))
    target = at + int(max(at * (RETELL_REEVAL_MIN_STEP_GROWTH - 1.0), RETELL_REEVAL_MIN_STEP_ABS))
    return (f"RNN steps >= {target} (now {cur_steps}), "
            f"or {RETELL_REEVAL_MIN_SECONDS // 3600}h elapsed, or new training stories")


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
