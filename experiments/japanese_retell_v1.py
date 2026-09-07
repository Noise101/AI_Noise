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
EVAL_REGIME = "narrative_order_recovery_v1"
SIGNIFICANT_TRAIN_GROWTH = 1.4
RETELL_REEVAL_MIN_STEP_GROWTH = 1.15   # RNN steps must grow this much, or ...
RETELL_REEVAL_MIN_SECONDS = 6 * 3600   # ... this long must pass, or new training
MAX_ORDER_PAIRS = 15                   # event pairs scored per held-out story


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


def _canonical_events(events: list) -> list:
    """Order- and key-stable view of one story's events for fingerprinting."""
    out = []
    for e in events:
        if isinstance(e, dict):
            out.append({"subject": e.get("subject", ""), "verb": e.get("verb", ""),
                        "obj": e.get("obj", "")})
        else:                                    # tuple/list (subject, verb, obj, ...)
            out.append({"subject": e[0] if len(e) > 0 else "",
                        "verb": e[1] if len(e) > 1 else "",
                        "obj": e[2] if len(e) > 2 else ""})
    return out


def _fingerprint(snapshot: list) -> str:
    """Hash the URL *and the event content* of every snapshot story, via
    canonical JSON -- so a snapshot whose events were changed or corrupted no
    longer hashes the same even when the URL list is untouched."""
    payload = json.dumps(
        sorted(({"url": s["url"], "events": _canonical_events(s["events"])} for s in snapshot),
               key=lambda s: s["url"]),
        sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


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
        if prev_steps and cur_steps >= prev_steps * RETELL_REEVAL_MIN_STEP_GROWTH:
            return True
        if time.time() - previous.get("evaluated_at", 0) >= RETELL_REEVAL_MIN_SECONDS:
            return True
        if not prev_steps:              # first time we ever see a model
            return True
        return False
    return False


def evaluate_retelling(stories: list[dict], previous: dict | None = None,
                       rnn_state: dict | None = None) -> dict:
    """Capability = NARRATIVE ORDER RECOVERY on a frozen, source-disjoint
    snapshot: the RNN-as-likelihood-model prefers the gold ordering of event
    pairs more often than a verb-position baseline built from the same training
    corpus.  Credited only after two independent significant measurements at
    growing training sizes.  The template round trip is a diagnostic.
    """
    previous = previous or {}
    regime_ok = previous.get("eval_regime") == EVAL_REGIME
    # eval-regime change: DO NOT inherit the old regime's streak / confirmation
    streak_prev = previous if regime_ok else {}
    snap = previous.get("test_snapshot") if regime_ok else None
    migrated = bool(previous.get("snapshot_migrated")) or (bool(previous) and not regime_ok)

    if snap:
        test = [{"url": s["url"], "events": s["events"]} for s in snap]
    else:
        train0 = [s for s in stories if not _held_out(s["url"]) and len(s.get("events", [])) >= 3]
        cand = [s for s in stories if _held_out(s["url"]) and len(s.get("events", [])) >= 3]
        if len(train0) < MIN_TRAIN_STORIES or len(cand) < MIN_TEST_STORIES:
            return {"version": 4, "status": "insufficient_stories", "eval_regime": EVAL_REGIME,
                    "train_stories": len(train0), "test_stories": len(cand),
                    "roundtrip_fidelity": None, "order_gain": None, "beats_baseline": False,
                    "significant_streak": 0, "recomputed": True,
                    "learning_curve": list(previous.get("learning_curve", [])) if regime_ok else []}
        test = [{"url": s["url"], "events": s["events"]} for s in cand]
        snap = test
        migrated = migrated or bool(previous)

    held_cols = {_collection(s["url"]) for s in snap}
    train = [s for s in stories if _collection(s["url"]) not in held_cols
             and len(s.get("events", [])) >= 3]
    n = len(test)
    fingerprint = _fingerprint(snap)
    has_model = bool(rnn_state and rnn_state.get("vocab"))
    cur_fp = (rnn_state or {}).get("model_fingerprint") if has_model else None
    cur_steps = (rnn_state or {}).get("steps_trained", 0) if has_model else 0

    # carry the last result forward unless the snapshot changed, the training
    # size changed, or the RNN both changed fingerprint AND crossed a re-eval
    # threshold.  "skipped" and "model unchanged" are reported separately.
    same_inputs = (regime_ok and previous.get("status") in ("measured", "no_generation_model")
                   and previous.get("train_stories") == len(train)
                   and previous.get("snapshot_fingerprint") == fingerprint
                   and bool(previous.get("order_gain") is not None) == has_model)
    if same_inputs and not _due_for_reeval(previous, rnn_state, len(train)):
        carried = dict(previous)
        carried["snapshot_migrated"] = migrated or bool(previous.get("snapshot_migrated"))
        carried["recomputed"] = False
        carried["rnn_model_changed"] = cur_fp != previous.get("rnn_fingerprint")
        carried["rnn_steps_now"] = cur_steps
        carried["next_reeval"] = _next_reeval_hint(previous, cur_steps)
        return carried

    # round-trip diagnostic (NOT a capability): template serialisation survives
    # its own generate -> re-parse round trip; shuffled loses the order signal.
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

    # the capability: narrative order recovery, RNN vs verb-position baseline
    mean_pos = _position_model(train)
    order_gain = z = rnn_acc = base_acc = None
    if has_model:
        scorer = _rnn_scorer(rnn_state)
        per_story = []
        r_accs, b_accs = [], []
        for s in test:
            got = _order_recovery(s["events"], scorer, mean_pos)
            if got is None:
                continue
            r, b, _m = got
            r_accs.append(r)
            b_accs.append(b)
            per_story.append(r - b)
        if len(per_story) >= MIN_TEST_STORIES:
            rnn_acc = sum(r_accs) / len(r_accs)
            base_acc = sum(b_accs) / len(b_accs)
            order_gain = sum(per_story) / len(per_story)
            var = sum((g - order_gain) ** 2 for g in per_story) / max(1, len(per_story) - 1)
            se = math.sqrt(var / len(per_story)) if var > 0 else 0.0
            z = order_gain / se if se > 0 else (99.0 if order_gain > 0 else 0.0)

    if z is None:
        status, significant = ("no_generation_model" if not has_model else "insufficient_scorable_stories"), False
    else:
        status, significant = "measured", (z >= SIGNIFICANCE_Z)
    last_sig_train = streak_prev.get("last_significant_train", 0)
    grew = len(train) >= last_sig_train * SIGNIFICANT_TRAIN_GROWTH
    streak = ((streak_prev.get("significant_streak", 0) + 1) if (significant and grew)
              else streak_prev.get("significant_streak", 0) if significant else 0)
    beats = significant and streak >= 2

    curve = list(streak_prev.get("learning_curve", []))
    point = {"train_stories": len(train), "eval_regime": EVAL_REGIME,
             "roundtrip_fidelity": round(mean_ordered, 3),
             "order_gain": None if order_gain is None else round(order_gain, 4),
             "gain_z": None if z is None else round(z, 2),
             "rnn_steps": cur_steps}
    if not curve or curve[-1].get("train_stories") != len(train) or curve[-1].get("rnn_steps") != cur_steps:
        curve.append(point)
    curve = curve[-200:]
    tail = [c.get("order_gain") or 0.0 for c in curve[-8:]]
    trend = "insufficient_data"
    if len(tail) >= 4:
        older = sum(tail[:len(tail) // 2]) / (len(tail) // 2)
        newer = sum(tail[len(tail) // 2:]) / (len(tail) - len(tail) // 2)
        trend = "improving" if newer > older + 0.005 else "declining" if newer < older - 0.005 else "flat"

    now = time.time()
    return {
        "version": 4, "status": status, "eval_regime": EVAL_REGIME,
        "snapshot_migrated": migrated, "recomputed": True,
        "regime_reset_from": previous.get("eval_regime") if (previous and not regime_ok) else None,
        "train_stories": len(train), "test_stories": n,
        "snapshot_stories": len(snap), "snapshot_fingerprint": fingerprint,
        "test_snapshot": snap,
        "roundtrip_fidelity": round(mean_ordered, 3),          # diagnostic, not capability
        "roundtrip_template_shuffled": round(mean_tmpl_shuffled, 3),
        "order_gain": None if order_gain is None else round(order_gain, 4),
        "rnn_pairwise_accuracy": None if rnn_acc is None else round(rnn_acc, 3),
        "position_baseline_accuracy": None if base_acc is None else round(base_acc, 3),
        "gain_z": None if z is None else round(z, 2),
        "beats_baseline_significant": significant,
        "beats_baseline": beats,
        "significant_streak": streak,
        "last_significant_train": len(train) if significant else last_sig_train,
        "rnn_fingerprint": cur_fp,
        "rnn_steps_at_eval": cur_steps,
        "rnn_steps_now": cur_steps,
        "rnn_model_changed": cur_fp != previous.get("rnn_fingerprint"),
        "evaluated_at": now,
        "next_reeval": _next_reeval_hint({"rnn_steps_at_eval": cur_steps, "evaluated_at": now}, cur_steps),
        "learning_curve": curve, "retelling_trend": trend,
        "limitations": ["capability = the RNN-as-likelihood-model recovering gold "
                        "event ORDER (pairwise) better than a verb-position baseline "
                        "on the FROZEN snapshot, at two growing training sizes. "
                        "free_retell is display only; the template round trip earns "
                        "nothing; an untrained RNN scores ~0.5 and cannot pass."],
    }


def _next_reeval_hint(previous: dict, cur_steps: int) -> str:
    target_steps = int(max(1, previous.get("rnn_steps_at_eval", cur_steps)) * RETELL_REEVAL_MIN_STEP_GROWTH)
    return (f"RNN steps >= {target_steps} (now {cur_steps}), "
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
