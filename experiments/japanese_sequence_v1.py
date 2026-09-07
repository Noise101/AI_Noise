#!/usr/bin/env python3
"""A from-scratch character RNN over Noise's Japanese reading.

Same tiny hand-written RNN as sequence_model_v1 (hidden 24, plain-list weights,
hand BPTT, no numpy) -- reused here on Japanese text so the developmental
reading loop gets a *continuous* capability signal (held-out bits/char) that
moves with a few hundred new sentences, and a `sample()` head toward
free-generation retelling (Phase 5c).

Text is the sentences of the books read so far (japanese_event_v1 keeps the
source sentence on every event).  Normalisation keeps kana, kanji and the core
punctuation; the vocabulary is frequency-capped so the pure-Python eval stays
affordable.

Capability credit is earned exactly as in sequence_model_v1: the per-source
paired improvement over the order-0 character baseline must clear a strict
one-sided z on a frozen, collection-disjoint split, for two consecutive cycles.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict

import urllib.parse

from sequence_model_v1 import TinyRNN, HIDDEN, SEQ_LEN, LEARNING_RATE

VERSION = 1
BENCHMARK_SALT = "japanese-sequence-benchmark:v1"
VOCAB_CAP = 200

# --- training identity ------------------------------------------------------
# re-audit #5 gave the RNN a training identity; re-audit #6 fixes it: the
# BOUNDARY fingerprint is now *only* the semantic training rules (model
# structure, normalisation, parser, provenance policy, split-policy version) --
# NOT the dynamic list of forbidden collections, which grows every time a new
# selection/final collection appears and used to wipe the RNN.  Instead the RNN
# accumulates `ever_trained_sources` (url -> normalised-text hash); it is retired
# ONLY when a collection it has ALREADY TRAINED ON becomes a held-out (forbidden)
# collection, or an already-trained source's text changes / disappears.  The
# character benchmark itself is DIAGNOSTIC only (re-audit #6 P2-1): it no longer
# claims a confirmed capability.
TRAINING_REGIME = "jseq_clean_v2"          # v2: boundary excludes forbidden_collections
# regimes whose weights transfer to the current one unchanged (only the identity
# bookkeeping was fixed): migrate in place, never retire for the code change.
COMPATIBLE_PREDECESSOR_REGIMES = ("jseq_clean_v1",)
SPLIT_POLICY_VERSION = "jseq_split_v1"     # bump if _split's held-out algorithm changes
PROVENANCE_POLICY = "fail_closed_heuristic_self_v1"
NORMALISATION_VERSION = 1
VOCAB_METHOD = "freq_capped_top200_min3"
MIN_TRAIN_CHARS = 3000
MIN_EVAL_CHARS = 1500
MAX_EVAL_CHARS = 20000
MIN_EVAL_SOURCES = 6
SIGNIFICANCE_Z = 3.0
DEFAULT_TRAIN_SECONDS = 6.0

_KEEP = re.compile(r"[ぁ-ゟ゠-ヿ一-鿿。、！？「」]")


def normalise(text: str) -> str:
    return "".join(_KEEP.findall(text))


def model_fingerprint(state: dict, steps_trained: int) -> str:
    """A stable digest of the RNN's identity: version, vocabulary, how much it has
    been trained, and a hash of the actual weights (rounded so float noise does
    not churn it).  Two calls agree iff the model would score text identically;
    it changes whenever the weights or step count move.  Downstream evaluators
    use this to tell 'the model has not changed' apart from 'I skipped the eval'.
    """
    if not state or not state.get("vocab"):
        return ""
    h = hashlib.sha256()
    h.update(f"seqrnn:v{VERSION}:steps={steps_trained}:".encode())
    h.update(("".join(state["vocab"])).encode())
    for name in ("Wxh", "Whh", "Why", "bh", "by"):
        w = state.get(name)
        if w is None:
            continue
        rows = w if w and isinstance(w[0], list) else [w]
        for row in rows:
            h.update(bytes(name, "ascii"))
            h.update(",".join(f"{v:.5f}" for v in row).encode())
    return h.hexdigest()[:16]


def _norm_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).encode()).hexdigest()[:16]


def boundary_fingerprint(ctx: dict) -> str:
    """The SEMANTIC training rules whose change makes old weights meaningless.
    Deliberately does NOT include the (dynamic) forbidden-collection list -- that
    grows as the corpus grows and is enforced per-cycle instead."""
    payload = json.dumps({
        "training_regime": TRAINING_REGIME,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "model_version": VERSION, "hidden_dim": HIDDEN, "seq_len": SEQ_LEN,
        "vocab_cap": VOCAB_CAP, "vocab_method": VOCAB_METHOD,
        "normalisation_version": NORMALISATION_VERSION,
        "provenance_policy": ctx.get("provenance_policy", PROVENANCE_POLICY),
        "parser_version": ctx.get("parser_version"),
        "read_only_training": bool(ctx.get("read_only", True)),
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _collection_of(url: str) -> str:
    from japanese_benchmark_v1 import collection
    return collection(url)


def training_data_fingerprint(ever_trained: dict, ctx: dict) -> dict:
    """Auditable identity: the semantic boundary plus the CUMULATIVE set of every
    source the model has ever trained on and its normalised-text hash."""
    sources = sorted((u, h) for u, h in ever_trained.items())
    src_payload = json.dumps(sources, ensure_ascii=False, separators=(",", ":"))
    cols = sorted({_collection_of(u) for u, _ in sources})
    return {
        "boundary_fingerprint": boundary_fingerprint(ctx),
        "ever_trained_set_fingerprint": hashlib.sha256(src_payload.encode()).hexdigest()[:16],
        "training_regime": TRAINING_REGIME,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "parser_version": ctx.get("parser_version"),
        "provenance_policy": ctx.get("provenance_policy", PROVENANCE_POLICY),
        "normalisation_version": NORMALISATION_VERSION,
        "vocab_method": VOCAB_METHOD,
        "model_structure": {"version": VERSION, "hidden": HIDDEN, "seq_len": SEQ_LEN},
        "ever_trained_source_count": len(sources),
        "ever_trained_collections": cols,
        "ever_trained_sources": [{"url": u, "text_hash": h} for u, h in sources],
    }


def _contamination_check(previous: dict, ctx: dict, corpus_texts: dict) -> dict:
    """Return {'reason': str|None, 'collisions': [...]}.  Retire ONLY when weights
    the model already holds are no longer valid: a different semantic boundary, a
    collection it TRAINED ON turned into a held-out set, or an already-trained
    source's text changed or vanished."""
    prev_state = previous.get("state") or {}
    if not prev_state or not prev_state.get("vocab"):
        return {"reason": None, "collisions": []}
    prev_regime = previous.get("training_regime") or prev_state.get("training_regime")
    if not prev_regime:
        return {"reason": "legacy_state_without_training_regime", "collisions": []}
    if prev_regime != TRAINING_REGIME and prev_regime not in COMPATIBLE_PREDECESSOR_REGIMES:
        return {"reason": f"training_regime_changed:{prev_regime}->{TRAINING_REGIME}", "collisions": []}
    # a compatible predecessor: skip the boundary check (the boundary format
    # itself changed) and skip the ledger checks (seeded below); just migrate.
    if prev_regime in COMPATIBLE_PREDECESSOR_REGIMES:
        return {"reason": None, "collisions": []}
    prev_boundary = (previous.get("training_data_fingerprint") or {}).get("boundary_fingerprint")
    if prev_boundary and prev_boundary != boundary_fingerprint(ctx):
        return {"reason": "training_boundary_changed", "collisions": [
            {"kind": "boundary", "was": prev_boundary, "now": boundary_fingerprint(ctx)}]}

    ever = dict(previous.get("ever_trained_sources") or {})
    if not ever:                                       # legacy-but-compatible: seed below, no retire
        return {"reason": None, "collisions": []}
    forbidden_cols = set(ctx.get("forbidden_collections", []))
    cur_hash = {u: _norm_hash(t) for u, t in corpus_texts.items()}
    collisions = []
    for url, h in ever.items():
        col = _collection_of(url)
        if col in forbidden_cols:
            collisions.append({"kind": "trained_collection_now_held_out", "url": url, "collection": col})
        elif url not in cur_hash:
            collisions.append({"kind": "trained_source_deleted", "url": url, "collection": col})
        elif cur_hash[url] != h:
            collisions.append({"kind": "trained_source_text_changed", "url": url,
                               "was": h, "now": cur_hash[url]})
    if collisions:
        kinds = sorted({c["kind"] for c in collisions})
        return {"reason": "trained_data_invalidated:" + ",".join(kinds), "collisions": collisions}
    return {"reason": None, "collisions": []}


def collection_key(url: str) -> str:
    """One key per work.  Aozora keeps the file path (an author is not one
    collection here -- character statistics don't leak between an author's
    stories the way meaning does); wikisource keeps the page title."""
    parsed = urllib.parse.urlsplit(url)
    path = urllib.parse.unquote(parsed.path).rstrip("/")
    if "/wiki/" in path:
        return f"{parsed.netloc}/wiki/{path.split('/wiki/', 1)[1]}"
    return f"{parsed.netloc}{path}" if path else url


def build_vocab(texts: dict[str, str]) -> list[str]:
    chars = Counter()
    for text in texts.values():
        chars.update(text)
    common = [ch for ch, n in chars.most_common(VOCAB_CAP) if n >= 3]
    return sorted(common) or sorted({ch for t in texts.values() for ch in t})[:VOCAB_CAP]


def _split(texts: dict[str, str], previous: dict) -> tuple[set[str], set[str]]:
    locked = previous.get("benchmark", {})
    if locked.get("held_out_collections"):
        held = set(locked["held_out_collections"])
    else:
        collections = sorted({collection_key(u) for u in texts},
                             key=lambda c: hashlib.sha256(f"{BENCHMARK_SALT}:{c}".encode()).hexdigest())
        held = set(collections[:max(1, len(collections) // 5)])
    return ({u for u in texts if collection_key(u) not in held},
            {u for u in texts if collection_key(u) in held})


def _insufficient(previous: dict, train_chars: int, held_chars: int) -> dict:
    return {"version": VERSION, "status": "insufficient_text",
            "benchmark": {"locked": False},
            "train_chars": train_chars, "held_out_chars": held_chars,
            "held_out_bits_per_char": None, "baseline_bits_per_char": None,
            "improvement_bits": None, "improvement_z": None,
            "held_out_sources_evaluated": 0, "beats_char_baseline": False,
            "significant_streak": 0, "capability_status": "diagnostic_only",
            "perplexity_trend": "not_yet_measured", "can_sample": False,
            "learning_curve": list(previous.get("learning_curve", [])), "samples": []}


def train_and_evaluate(raw_texts: dict[str, str], previous: dict | None = None,
                       train_seconds: float = DEFAULT_TRAIN_SECONDS,
                       max_steps: int | None = None,
                       training_context: dict | None = None) -> dict:
    previous = dict(previous or {})
    ctx = dict(training_context or {})
    ctx.setdefault("provenance_policy", PROVENANCE_POLICY)
    ctx.setdefault("read_only", True)
    texts = {u: n for u, t in raw_texts.items() if len(n := normalise(t)) >= 40}

    # --- contamination gate: retire ONLY when weights the model already holds
    # are invalid (boundary changed, a trained collection became held-out, or a
    # trained source's text changed / disappeared).  A new held-out collection it
    # never trained on is NOT a reason. ---
    chk = _contamination_check(previous, ctx, raw_texts)
    reset_reason, collisions = chk["reason"], chk["collisions"]
    retired = None
    contamination_status = "clean_continued"
    prev_ever = dict(previous.get("ever_trained_sources") or {})
    if reset_reason:
        retired = {"retired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "reset_reason": reset_reason, "collisions": collisions[:50],
                   "retired_model_fingerprint": previous.get("model_fingerprint"),
                   "retired_steps_trained": previous.get("steps_trained", 0),
                   "retired_training_regime": previous.get("training_regime"),
                   "retired_ever_trained_source_count": len(prev_ever),
                   "retired_report": {k: previous.get(k) for k in
                                      ("status", "held_out_bits_per_char", "baseline_bits_per_char",
                                       "improvement_z", "steps_trained", "learning_curve")},
                   "retired_state": previous.get("state")}
        parent_fp = previous.get("model_fingerprint")
        _prior = list(previous.get("retirement_log", []))
        previous = {"learning_curve": [], "retirement_log": _prior}
        prev_ever = {}
        contamination_status = "retired_replaced"
    else:
        parent_fp = previous.get("parent_model_fingerprint")

    train_urls, held_urls = _split(texts, previous)
    train_chars = sum(len(texts[u]) for u in train_urls)
    held_chars = sum(len(texts[u]) for u in held_urls)

    # compatible-predecessor migration: a jseq_clean_v1 state carries the same
    # weights but no cumulative ledger.  Adopt its recorded training_sources as
    # the initial ever_trained set, upgrade the regime, keep the weights + steps,
    # do NOT retire (re-audit #6: "don't retire the clean RNN for the code change").
    boundary_migrated = False
    if (not reset_reason and not prev_ever and previous.get("state", {}).get("vocab")
            and (previous.get("training_regime") in COMPATIBLE_PREDECESSOR_REGIMES
                 or (previous.get("training_data_fingerprint") or {}).get("training_sources"))):
        legacy = ((previous.get("training_data_fingerprint") or {}).get("training_sources")
                  or previous.get("state", {}).get("ever_trained_sources") or [])
        if isinstance(legacy, dict):
            prev_ever = dict(legacy)
        else:
            prev_ever = {s["url"]: s["text_hash"] for s in legacy if isinstance(s, dict) and s.get("url")}
        boundary_migrated = True

    started_clean_at = (previous.get("started_clean_at")
                        or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    retirement_log = list(previous.get("retirement_log", []))
    if retired:
        retirement_log = retirement_log + [{k: retired[k] for k in retired if k != "retired_state"}]

    def _ident(ever_trained: dict, added: int, dropped: list) -> dict:
        tdf = training_data_fingerprint(ever_trained, ctx)
        return {"training_regime": TRAINING_REGIME,
                "split_policy_version": SPLIT_POLICY_VERSION,
                "training_data_fingerprint": tdf,
                "ever_trained_sources": ever_trained,
                "ever_trained_source_count": len(ever_trained),
                "ever_trained_collections": tdf["ever_trained_collections"],
                "sources_added_this_cycle": added,
                "dropped_trained_sources": dropped,
                "read_only_training": bool(ctx.get("read_only")),
                "contamination_status": contamination_status,
                "reset_reason": reset_reason, "reset_collisions": collisions[:50],
                "boundary_migrated": boundary_migrated,
                "started_clean_at": started_clean_at,
                "parent_model_fingerprint": parent_fp,
                "retired_model": retired,
                "retirement_log": retirement_log}

    dropped_now = sorted(u for u in prev_ever if u not in {n for n in raw_texts})
    if train_chars < MIN_TRAIN_CHARS or held_chars < MIN_EVAL_CHARS:
        return {**_insufficient(previous, train_chars, held_chars),
                **_ident(prev_ever, 0, dropped_now),
                "steps_trained": previous.get("steps_trained", 0)}

    vocab = previous.get("state", {}).get("vocab") or build_vocab(texts)
    model = TinyRNN(vocab, previous.get("state"))
    rng = random.Random(previous.get("steps_trained", 0) + 1)

    windows: list[str] = []
    for url in sorted(train_urls):
        text = texts[url]
        for start in range(0, max(1, len(text) - SEQ_LEN), SEQ_LEN):
            windows.append(text[start:start + SEQ_LEN + 1])
    rng.shuffle(windows)

    deadline = time.monotonic() + train_seconds
    steps = 0
    loss_sum = 0.0
    for window in windows:
        if (max_steps is not None and steps >= max_steps) or time.monotonic() > deadline:
            break
        loss_sum += model.train_step(window, LEARNING_RATE)
        steps += 1
    steps_trained = previous.get("steps_trained", 0) + steps

    base_counts = Counter()
    for url in train_urls:
        base_counts.update(texts[url])
    base_total = sum(base_counts.values()) or 1
    log_base = {ch: math.log(max(base_counts.get(ch, 0.5) / base_total, 1e-12)) for ch in vocab}

    per_source: list[float] = []
    model_nll = base_nll = evaluated = 0.0
    for url in sorted(held_urls):
        if evaluated >= MAX_EVAL_CHARS:
            break
        text = texts[url][:MAX_EVAL_CHARS]
        m_bpc, n = model.bits_per_char(text)
        if n < 40:
            continue
        b_nll = sum(-log_base.get(ch, math.log(1e-12))
                    for a, ch in zip(text, text[1:]) if a in model.index and ch in model.index)
        b_bpc = b_nll / n / math.log(2)
        per_source.append(b_bpc - m_bpc)
        model_nll += m_bpc * n
        base_nll += b_bpc * n
        evaluated += n
    model_bpc = model_nll / evaluated if evaluated else 0.0
    baseline_bpc = base_nll / evaluated if evaluated else 0.0

    n_sources = len(per_source)
    mean_gain = sum(per_source) / n_sources if n_sources else 0.0
    if n_sources >= 2:
        var = sum((d - mean_gain) ** 2 for d in per_source) / (n_sources - 1)
        se = math.sqrt(var / n_sources) if var > 0 else 0.0
        z = mean_gain / se if se > 0 else (99.0 if mean_gain > 0 else 0.0)
    else:
        se = z = 0.0
    p = round(0.5 * math.erfc(z / math.sqrt(2)), 6) if z > 0 else 1.0

    # DIAGNOSTIC ONLY (re-audit #6 P2-1): the character benchmark is a learning
    # signal, never a confirmed capability.  `significant_streak` is a plain
    # counter for the curve; there is no "beats_char_baseline" pass.
    significant_now = (n_sources >= MIN_EVAL_SOURCES and z >= SIGNIFICANCE_Z
                       and evaluated >= MIN_EVAL_CHARS)
    streak = previous.get("significant_streak", 0) + 1 if significant_now else 0

    # cumulative training ledger: add every source actually trained on this cycle
    ever_trained = dict(prev_ever)
    added = 0
    for u in train_urls:
        h = _norm_hash(raw_texts.get(u, texts.get(u, "")))
        if ever_trained.get(u) != h:
            added += 1
        ever_trained[u] = h
    ident = _ident(ever_trained, added, dropped_now)

    final_state = model.state()
    fp = model_fingerprint(final_state, steps_trained)
    ident["model_fingerprint"] = fp

    curve = list(previous.get("learning_curve", []))
    point = {"steps_trained": steps_trained, "train_chars": train_chars,
             "held_out_bits_per_char": round(model_bpc, 4),
             "baseline_bits_per_char": round(baseline_bpc, 4),
             "improvement_bits": round(mean_gain, 4), "improvement_z": round(z, 3)}
    if not curve or curve[-1]["steps_trained"] != steps_trained:
        curve.append(point)
    curve = curve[-200:]
    tail = [c["held_out_bits_per_char"] for c in curve[-8:]]
    trend = "insufficient_data"
    if len(tail) >= 4:
        older = sum(tail[:len(tail) // 2]) / (len(tail) // 2)
        newer = sum(tail[len(tail) // 2:]) / (len(tail) - len(tail) // 2)
        trend = "improving" if newer < older - 0.01 else "declining" if newer > older + 0.01 else "flat"

    return {
        "version": VERSION,
        "capability_status": "diagnostic_only",
        "status": ("improving_not_a_capability_claim" if mean_gain > 0
                   else "below_char_baseline"),
        "benchmark": {"locked": True, "kind": "diagnostic",
                      "held_out_collections": sorted({collection_key(u) for u in held_urls}),
                      "train_sources": len(train_urls), "held_out_sources": len(held_urls)},
        "hidden_dim": HIDDEN, "vocab_size": len(vocab),
        "train_chars": train_chars, "held_out_chars": held_chars,
        "steps_this_cycle": steps, "steps_trained": steps_trained,
        "mean_train_loss": round(loss_sum / steps, 4) if steps else None,
        "held_out_bits_per_char": round(model_bpc, 4),
        "baseline_bits_per_char": round(baseline_bpc, 4),
        "improvement_bits": round(mean_gain, 4),
        "held_out_sources_evaluated": n_sources,
        "improvement_z": round(z, 3), "improvement_p_one_sided": p,
        "significant_streak": streak,
        "improvement_significant_now": significant_now,   # diagnostic, not a pass
        "beats_char_baseline": False,                     # this benchmark cannot confer capability
        "perplexity_trend": trend, "can_sample": True,
        "learning_curve": curve,
        "samples": [generate(model, "むかしむかし", 90),
                    generate(model, "おじいさんは", 90)],
        "model_fingerprint": fp,
        "state": {**final_state, "version": VERSION, "steps_trained": steps_trained,
                  "model_fingerprint": fp, "training_regime": TRAINING_REGIME,
                  "ever_trained_sources": ever_trained,
                  "boundary_fingerprint": ident["training_data_fingerprint"]["boundary_fingerprint"]},
        **ident,
        "limitations": ["character-level, DIAGNOSTIC ONLY -- never a confirmed capability. "
                        "Retired only when a trained collection becomes held-out or a "
                        "trained source's text changes/disappears; a new held-out "
                        "collection it never trained on does not retire it."],
    }


def generate(model: TinyRNN, prime: str, length: int = 120,
             temperature: float = 0.8, rng: random.Random | None = None) -> str:
    """Sampling loop that keeps Japanese text (TinyRNN.sample normalises to ASCII)."""
    rng = rng or random.Random(0)
    h = [0.0] * HIDDEN
    last = None
    for ch in normalise(prime) or "むかし":
        xi = model.index.get(ch)
        if xi is not None:
            h, _ = model._step(xi, h)
            last = xi
    if last is None:
        last = 0
    out = []
    for _ in range(length):
        h, probs = model._step(last, h)
        if temperature != 1.0:
            logits = [math.log(max(pr, 1e-12)) / temperature for pr in probs]
            m = max(logits)
            exps = [math.exp(v - m) for v in logits]
            total = sum(exps)
            probs = [v / total for v in exps]
        r = rng.random()
        cum = 0.0
        choice = len(probs) - 1
        for idx, pr in enumerate(probs):
            cum += pr
            if r <= cum:
                choice = idx
                break
        out.append(model.vocab[choice])
        last = choice
    return "".join(out)


def main() -> None:
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path,
                        default=Path(__file__).resolve().parent.parent / ".local")
    parser.add_argument("--seconds", type=float, default=DEFAULT_TRAIN_SECONDS)
    args = parser.parse_args()
    events = json.loads((args.runtime / "reading-events.json").read_text(encoding="utf-8"))
    curr = json.loads((args.runtime / "reading-curriculum.json").read_text(encoding="utf-8"))
    texts: dict[str, str] = defaultdict(str)
    for bid, evs in events.items():
        url = curr.get("shelf", {}).get(bid, {}).get("url", f"book:{bid}")
        texts[url] += "".join(e.get("sentence", "") for e in evs)
    out = args.runtime / "reading-sequence.json"
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    report = train_and_evaluate(dict(texts), previous, args.seconds)
    out.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in
                      ("status", "held_out_bits_per_char", "baseline_bits_per_char",
                       "improvement_bits", "improvement_z", "beats_char_baseline",
                       "steps_trained")}, ensure_ascii=False))
    for s in report.get("samples", []):
        print(" ", s)


if __name__ == "__main__":
    main()
