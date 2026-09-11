#!/usr/bin/env python3
"""Small, local, online associative memory for direct conversation.

Not a language model, not a source of truth.  It learns compact topic vectors
by predicting which OTHER topics/content-words Noise's own conversation
episodes (`noise_chat_v1`'s claims and topic history) place near each other.
When a new, unknown topic comes up, the nearest already-discussed topic is a
*recall prompt* -- "did you mean the thing we talked about before" -- never an
assertion that the two are the same, and it never changes a belief or a
stored claim by itself.  Same evidence-score-0 boundary and the same
dependency-free skip-gram-with-negative-sampling implementation as
`semantic_representation_v1` (reading-side); this module is conversation-side
and trains on conversation state instead of read events.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

VERSION = 1
DIM = 12
MAX_TOPICS = 500
MAX_CONTEXTS = 800
MAX_PAIRS_PER_TURN_BATCH = 200
NEGATIVES = 2
BASE_LR = 0.05
MIN_SIMILARITY = 0.15
MIN_SUPPORT = 1                # a single shared conversation is enough --
                                # conversation vocabulary is tiny, unlike a
                                # book corpus; this is a recall prompt, not a
                                # capability claim, so a low bar is fine

_KANJI_RUN = re.compile(r"[一-鿿々]+")
_KATA_RUN = re.compile(r"[゠-ヿー]{2,}")


def _content_terms(text: str) -> set[str]:
    return (set(_KANJI_RUN.findall(text or "")) | set(_KATA_RUN.findall(text or "")))


def _hid(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _randvec(key: str, dim: int = DIM) -> list[float]:
    raw = hashlib.shake_256(key.encode()).digest(dim * 2)
    return [((int.from_bytes(raw[i * 2:i * 2 + 2], "big") / 65535.0) - 0.5) * 0.1
            for i in range(dim)]


def _sigmoid(x: float) -> float:
    x = max(-12.0, min(12.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(_dot(a, a)), math.sqrt(_dot(b, b))
    return _dot(a, b) / (na * nb) if na > 1e-9 and nb > 1e-9 else 0.0


def _blank() -> dict:
    return {"version": VERSION, "word_vectors": {}, "context_vectors": {},
            "word_counts": {}, "context_counts": {}, "negative_pool": [],
            "processed_turns": [], "pairs_seen": 0, "updates": 0}


def _claim_pairs(claims: dict) -> list[tuple[str, str]]:
    """subject -> shared-content-word context.  This is a real skip-gram
    pair: when TWO DIFFERENT subjects are both told to have the same content
    word ("献は道具です" / "鍵盤は道具です"), both get pulled toward the SAME
    context vector ("claim:道具") and so, transitively, toward each other --
    the standard distributional-similarity effect. A single occurrence has no
    such shared attractor and legitimately produces no similarity."""
    pairs: list[tuple[str, str]] = []
    for subject, entries in (claims or {}).items():
        if not _content_terms(subject):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("retracted"):
                continue
            for term in _content_terms(entry.get("predicate", "")):
                if term != subject:
                    pairs.append((subject, f"claim:{term}"))
    return pairs


def _proximity_pairs(topic_stack: list[str]) -> list[tuple[str, str]]:
    """topic <-> topic pairs from being discussed near each other.  Unlike
    claim content, a conversation rarely mentions the same two topics
    together often enough to build a shared-context attractor, so these are
    trained as a DIRECT symmetric pull between the two topics' own vectors
    (see `_symmetric_update`), not through an intermediate context vector."""
    pairs: list[tuple[str, str]] = []
    stack = [t for t in (topic_stack or []) if _content_terms(t)]
    for i, topic in enumerate(stack):
        for other in stack[max(0, i - 3):i] + stack[i + 1:i + 4]:
            if other != topic:
                pairs.append((topic, other))
    return pairs


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


def _symmetric_update(state: dict, a: str, b: str, label: float, lr: float) -> None:
    """Pull (label=1) or push apart (label=0) two topics' OWN vectors
    directly -- see `_proximity_pairs` for why this differs from `_update`."""
    va = state["word_vectors"].setdefault(a, _randvec("w:" + a))
    vb = state["word_vectors"].setdefault(b, _randvec("w:" + b))
    pred = _sigmoid(_dot(va, vb))
    grad = label - pred
    old_a = list(va)
    for i in range(DIM):
        va[i] = max(-2.0, min(2.0, va[i] + lr * grad * vb[i]))
        vb[i] = max(-2.0, min(2.0, vb[i] + lr * grad * old_a[i]))
    state["updates"] += 1


def _prune(state: dict) -> None:
    if len(state["word_vectors"]) > MAX_TOPICS:
        keep = {w for w, _ in Counter(state["word_counts"]).most_common(MAX_TOPICS)}
        state["word_vectors"] = {w: v for w, v in state["word_vectors"].items() if w in keep}
        state["word_counts"] = {w: n for w, n in state["word_counts"].items() if w in keep}
    if len(state["context_vectors"]) > MAX_CONTEXTS:
        keep = {c for c, _ in Counter(state["context_counts"]).most_common(MAX_CONTEXTS)}
        state["context_vectors"] = {c: v for c, v in state["context_vectors"].items() if c in keep}
        state["context_counts"] = {c: n for c, n in state["context_counts"].items() if c in keep}
        state["negative_pool"] = [c for c in state.get("negative_pool", []) if c in keep]


def _topic_negatives(state: dict, positive: str, exclude: str, salt: str) -> list[str]:
    keys = [w for w in (state.get("word_vectors") or {}) if w not in (positive, exclude)]
    if not keys:
        return []
    out = []
    start = int(_hid(f"tneg|{salt}"), 16) % len(keys)
    stride = 1 + int(_hid(f"tstride|{salt}"), 16) % max(1, len(keys) - 1)
    for i in range(len(keys)):
        key = keys[(start + i * stride) % len(keys)]
        if key not in out:
            out.append(key)
            if len(out) >= NEGATIVES:
                break
    return out


def learn(previous: dict | None, claims: dict, topic_stack: list[str],
         turn_id: str | None = None) -> dict:
    """Incremental, bounded update from the CURRENT conversation state only --
    a claim already learned from is not replayed (processed_turns), so one
    remembered fact does not become thousands of fake repeated observations."""
    state = dict(previous or {})
    if state.get("version") != VERSION:
        state = _blank()
    for k, v in _blank().items():
        state.setdefault(k, v)
    if turn_id:
        if turn_id in state["processed_turns"]:
            return state
        state["processed_turns"] = (state["processed_turns"] + [turn_id])[-2000:]

    for idx, (word, context) in enumerate(_claim_pairs(claims)[:MAX_PAIRS_PER_TURN_BATCH]):
        if word not in state["word_vectors"] and len(state["word_vectors"]) >= MAX_TOPICS:
            continue
        if context not in state["context_vectors"] and len(state["context_vectors"]) >= MAX_CONTEXTS:
            continue
        state["word_counts"][word] = state["word_counts"].get(word, 0) + 1
        state["context_counts"][context] = state["context_counts"].get(context, 0) + 1
        lr = max(0.01, BASE_LR / math.sqrt(1.0 + state["pairs_seen"] / 500.0))
        _update(state, word, context, 1.0, lr)
        for neg in _negative_keys(state, context, f"{turn_id or ''}|c{idx}"):
            _update(state, word, neg, 0.0, lr)
        state["pairs_seen"] += 1

    for idx, (a, b) in enumerate(_proximity_pairs(topic_stack)[:MAX_PAIRS_PER_TURN_BATCH]):
        if a not in state["word_vectors"] and len(state["word_vectors"]) >= MAX_TOPICS:
            continue
        if b not in state["word_vectors"] and len(state["word_vectors"]) >= MAX_TOPICS:
            continue
        state["word_counts"][a] = state["word_counts"].get(a, 0) + 1
        state["word_counts"][b] = state["word_counts"].get(b, 0) + 1
        lr = max(0.01, BASE_LR / math.sqrt(1.0 + state["pairs_seen"] / 500.0))
        _symmetric_update(state, a, b, 1.0, lr)
        for neg in _topic_negatives(state, a, b, f"{turn_id or ''}|p{idx}"):
            _symmetric_update(state, a, neg, 0.0, lr)
        state["pairs_seen"] += 1

    _prune(state)
    return state


def recall(topic: str, state: dict | None, known_topics: "set[str] | None" = None,
          top_k: int = 1) -> list[dict]:
    """Nearest previously-discussed topics to `topic` -- a bounded PROPOSAL for
    a recall prompt, never a claim that the two are the same thing. Restricted
    to `known_topics` (topics actually in `claims`/`topic_stack`, i.e. things
    Noise could truthfully say "we talked about") when given."""
    if not state or topic not in (state.get("word_vectors") or {}):
        return []
    target = state["word_vectors"][topic]
    out = []
    for other, vec in state["word_vectors"].items():
        if other == topic:
            continue
        if known_topics is not None and other not in known_topics:
            continue
        if state.get("word_counts", {}).get(other, 0) < MIN_SUPPORT:
            continue
        sim = _cosine(target, vec)
        if sim >= MIN_SIMILARITY:
            out.append({"topic": other, "similarity": round(sim, 3)})
    out.sort(key=lambda r: -r["similarity"])
    return out[:top_k]
