#!/usr/bin/env python3
"""Small, local, online semantic representation learned from Noise's events.

This is deliberately not a language model and not a source of truth.  It learns
compact word vectors by predicting the *usage contexts* that occurred in
Noise's own ``heuristic_self`` event stream.  A word's coarse-class proposal is
then a weighted vote from nearby words that the ordinary belief machinery has
already grounded.  The proposal can be tried by the cognitive controller, but
it never changes a belief and never confers capability by itself.

The implementation is dependency-free skip-gram with negative sampling.  Work
is bounded per reading cycle and source fingerprints prevent a reread from
being counted as a new experience.  A deterministic held-out-anchor diagnostic
is reported only as a diagnostic; the real capability gate remains
``capability_probe_v1``'s unbiased selection + unopened final.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter

VERSION = 1
DIM = 20
BOOKS_PER_CYCLE = 8
MAX_WORDS = 4000
MAX_CONTEXTS = 6000
MAX_PAIRS_PER_BOOK = 320
NEGATIVES = 2
BASE_LR = 0.035
MIN_ANCHORS = 3
MIN_SIMILARITY = 0.08
DIAGNOSTIC_MOD = 5


def enabled() -> bool:
    return os.environ.get("AI_NOISE_SEMANTIC") != "0"


def _hid(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fingerprint(events: list[dict]) -> str:
    rows = [f"{e.get('subject','')}|{e.get('verb','')}|{e.get('obj','')}"
            for e in events if isinstance(e, dict)
            and e.get("provenance") == "heuristic_self"]
    return _hid("\n".join(rows))[:16]


def _real_word(word: str) -> bool:
    word = (word or "").strip()
    if not (2 <= len(word) <= 8) or any(c in word for c in " 、。！？「」『』"):
        return False
    try:
        from japanese_word_meaning_v1 import _is_wordlike
        return _is_wordlike(word)
    except Exception:
        return True


def _real_verb(verb: str) -> bool:
    verb = (verb or "").strip()
    if not (2 <= len(verb) <= 10):
        return False
    try:
        from cognition_v1 import _informative_verb
        return _informative_verb(verb)
    except Exception:
        return True


def _randvec(key: str, dim: int = DIM) -> list[float]:
    """Stable small initial vector without a global RNG or Python hash()."""
    raw = hashlib.shake_256(key.encode()).digest(dim * 2)
    return [((int.from_bytes(raw[i * 2:i * 2 + 2], "big") / 65535.0) - 0.5) * 0.1
            for i in range(dim)]


def _sigmoid(x: float) -> float:
    x = max(-12.0, min(12.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(_dot(a, a))
    nb = math.sqrt(_dot(b, b))
    return _dot(a, b) / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0


def _blank(parser_version: int) -> dict:
    return {"version": VERSION, "parser_version": parser_version,
            "word_vectors": {}, "context_vectors": {}, "word_counts": {},
            "context_counts": {}, "negative_pool": [],
            "processed_sources": {}, "source_cursor": 0,
            "pairs_seen": 0, "updates": 0, "cycles": 0, "learning_curve": []}


def _event_pairs(events: list[dict]) -> list[tuple[str, str]]:
    """Observed word -> usage-feature pairs.  No genus/dictionary labels."""
    pairs: list[tuple[str, str]] = []
    for e in events:
        if not isinstance(e, dict) or e.get("provenance") != "heuristic_self":
            continue
        s = (e.get("subject") or "").strip()
        v = (e.get("verb") or "").strip()
        o = (e.get("obj") or "").strip()
        if _real_word(s) and _real_verb(v):
            pairs.extend(((s, "role:subject"), (s, f"verb:{v}")))
            if _real_word(o):
                pairs.extend(((s, f"co:{o}"), (o, "role:object"),
                              (o, f"verb:{v}"), (o, f"co:{s}")))
        elif _real_word(o) and _real_verb(v):
            pairs.extend(((o, "role:object"), (o, f"verb:{v}")))
    # repeated mentions are evidence, but bound one pathological book
    return pairs[:MAX_PAIRS_PER_BOOK]


def _negative_keys(state: dict, positive: str, salt: str) -> list[str]:
    keys = state.get("negative_pool") or []
    if len(keys) <= 1:
        return []
    out = []
    start = int(_hid(f"neg|{salt}"), 16) % len(keys)
    stride = 1 + int(_hid(f"stride|{salt}"), 16) % max(1, len(keys) - 1)
    for i in range(len(keys)):
        key = keys[(start + i * stride) % len(keys)]
        if key != positive and key not in out:
            out.append(key)
            if len(out) >= NEGATIVES:
                break
    return out


def _update(state: dict, word: str, context: str, label: float, lr: float) -> None:
    wv = state["word_vectors"].setdefault(word, _randvec("w:" + word))
    if context not in state["context_vectors"]:
        state["context_vectors"][context] = _randvec("c:" + context)
        state.setdefault("negative_pool", []).append(context)
    cv = state["context_vectors"][context]
    pred = _sigmoid(_dot(wv, cv))
    grad = label - pred
    old_w = list(wv)
    for i in range(DIM):
        wv[i] = max(-2.0, min(2.0, wv[i] + lr * grad * cv[i]))
        cv[i] = max(-2.0, min(2.0, cv[i] + lr * grad * old_w[i]))
    state["updates"] += 1


def _prune(state: dict) -> None:
    if len(state["word_vectors"]) > MAX_WORDS:
        keep = {w for w, _ in Counter(state["word_counts"]).most_common(MAX_WORDS)}
        state["word_vectors"] = {w: v for w, v in state["word_vectors"].items() if w in keep}
        state["word_counts"] = {w: n for w, n in state["word_counts"].items() if w in keep}
    if len(state["context_vectors"]) > MAX_CONTEXTS:
        keep = {c for c, _ in Counter(state["context_counts"]).most_common(MAX_CONTEXTS)}
        state["context_vectors"] = {c: v for c, v in state["context_vectors"].items() if c in keep}
        state["context_counts"] = {c: n for c, n in state["context_counts"].items() if c in keep}
        state["negative_pool"] = [c for c in state.get("negative_pool", []) if c in keep]


def _anchors(wm_state: dict, exclude_holdout: bool = False) -> list[tuple[str, str, float]]:
    out = []
    for word, belief in (wm_state.get("beliefs") or {}).items():
        if not (_real_word(word) and belief.get("understood") and belief.get("genus")):
            continue
        if float(belief.get("confidence", 0.0)) < 0.6:
            continue
        if exclude_holdout and int(_hid("semantic-diag:" + word), 16) % DIAGNOSTIC_MOD == 0:
            continue
        out.append((word, belief["genus"], float(belief.get("confidence", 0.6))))
    return out


def infer_genus(word: str, state: dict | None, wm_state: dict,
                exclude_self: bool = True, diagnostic: bool = False) -> dict:
    """Nearest grounded-anchor vote.  A proposal, never a belief mutation."""
    if not state or word not in (state.get("word_vectors") or {}):
        return {"genus": "", "confidence": 0.0, "support": 0, "neighbours": []}
    target = state["word_vectors"][word]
    neighbours = []
    for other, genus, conf in _anchors(wm_state, exclude_holdout=diagnostic):
        if (exclude_self and other == word) or other not in state["word_vectors"]:
            continue
        sim = _cosine(target, state["word_vectors"][other])
        if sim >= MIN_SIMILARITY:
            neighbours.append((sim, other, genus, conf))
    neighbours.sort(reverse=True)
    neighbours = neighbours[:12]
    votes: Counter = Counter()
    counts: Counter = Counter()
    for sim, _other, genus, conf in neighbours:
        votes[genus] += sim * sim * conf
        counts[genus] += 1
    if not votes:
        return {"genus": "", "confidence": 0.0, "support": 0, "neighbours": []}
    ranked = votes.most_common(2)
    genus, strength = ranked[0]
    support = counts[genus]
    margin = strength - (ranked[1][1] if len(ranked) > 1 else 0.0)
    total = sum(votes.values())
    confidence = (strength / total) * min(1.0, support / 5.0) if total else 0.0
    if support < MIN_ANCHORS or margin <= 0.03:
        genus, confidence = "", 0.0
    return {"genus": genus, "confidence": round(min(0.65, confidence), 3),
            "support": support,
            "neighbours": [{"word": o, "genus": g, "similarity": round(s, 3)}
                           for s, o, g, _c in neighbours[:5]]}


def _diagnostic(state: dict, wm_state: dict) -> dict:
    beliefs = wm_state.get("beliefs") or {}
    test = [w for w, b in beliefs.items()
            if b.get("understood") and b.get("genus") and w in state["word_vectors"]
            and int(_hid("semantic-diag:" + w), 16) % DIAGNOSTIC_MOD == 0]
    genera = Counter(b["genus"] for w, b in beliefs.items()
                     if b.get("understood") and b.get("genus")
                     and int(_hid("semantic-diag:" + w), 16) % DIAGNOSTIC_MOD != 0)
    baseline = genera.most_common(1)[0][0] if genera else ""
    correct = base_correct = covered = 0
    for word in test:
        proposal = infer_genus(word, state, wm_state, diagnostic=True)
        if proposal["genus"]:
            covered += 1
            correct += int(proposal["genus"] == beliefs[word]["genus"])
        base_correct += int(baseline == beliefs[word]["genus"])
    n = len(test)
    return {"status": "diagnostic_only", "test_words": n, "covered": covered,
            "coverage": round(covered / n, 3) if n else 0.0,
            "correct": correct, "baseline_correct": base_correct,
            "accuracy": round(correct / n, 3) if n else 0.0,
            "baseline_accuracy": round(base_correct / n, 3) if n else 0.0,
            "note": "anchor holdout diagnostic; never confers capability"}


def learn(events_store: dict[str, list[dict]], wm_state: dict,
          previous: dict | None, parser_version: int, cycle: int) -> dict:
    previous = previous or {}
    reset_reason = None
    if previous.get("version") != VERSION:
        reset_reason = "semantic_version_changed" if previous else None
        state = _blank(parser_version)
    elif previous.get("parser_version") != parser_version:
        reset_reason = (f"parser_version_changed:{previous.get('parser_version')}->{parser_version}")
        state = _blank(parser_version)
    else:
        state = dict(previous)
        for k, v in _blank(parser_version).items():
            state.setdefault(k, v)
        # migration from a prerelease state written before negative_pool existed
        if not state.get("negative_pool") and state.get("context_vectors"):
            state["negative_pool"] = list(state["context_vectors"])

    pending = []
    for source, events in events_store.items():
        fp = _fingerprint(events)
        if fp and state["processed_sources"].get(source) != fp:
            pending.append((source, fp, events))
    pending.sort(key=lambda x: _hid(f"{cycle}|{x[0]}"))
    trained_sources = 0
    positive_pairs = 0
    for source, fp, events in pending[:BOOKS_PER_CYCLE]:
        pairs = _event_pairs(events)
        for idx, (word, context) in enumerate(pairs):
            if word not in state["word_vectors"] and len(state["word_vectors"]) >= MAX_WORDS:
                continue
            if context not in state["context_vectors"] and len(state["context_vectors"]) >= MAX_CONTEXTS:
                continue
            state["word_counts"][word] = state["word_counts"].get(word, 0) + 1
            state["context_counts"][context] = state["context_counts"].get(context, 0) + 1
            lr = max(0.008, BASE_LR / math.sqrt(1.0 + state["pairs_seen"] / 5000.0))
            _update(state, word, context, 1.0, lr)
            for neg in _negative_keys(state, context, f"{source}|{idx}"):
                _update(state, word, neg, 0.0, lr)
            state["pairs_seen"] += 1
            positive_pairs += 1
        state["processed_sources"][source] = fp
        trained_sources += 1
    _prune(state)
    state["cycles"] += 1
    diag = _diagnostic(state, wm_state)
    model_fp = _hid("|".join(
        f"{w}:{','.join(f'{x:.3f}' for x in v[:4])}"
        for w, v in sorted(state["word_vectors"].items())))[:16]
    point = {"cycle": cycle, "sources": len(state["processed_sources"]),
             "words": len(state["word_vectors"]), "pairs": state["pairs_seen"],
             "accuracy": diag["accuracy"], "baseline_accuracy": diag["baseline_accuracy"],
             "coverage": diag["coverage"]}
    curve = list(state.get("learning_curve", []))
    if not curve or curve[-1].get("cycle") != cycle:
        curve.append(point)
    state["learning_curve"] = curve[-200:]
    return {**state, "status": "learning" if pending else "caught_up",
            "sources_trained_this_cycle": trained_sources,
            "positive_pairs_this_cycle": positive_pairs,
            "pending_sources": max(0, len(pending) - trained_sources),
            "word_count": len(state["word_vectors"]),
            "context_count": len(state["context_vectors"]),
            "model_fingerprint": model_fp, "diagnostic": diag,
            "reset_reason": reset_reason,
            "note": "small predictive usage embeddings; proposals only, never truth"}
