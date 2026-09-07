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
            "beats_char_baseline_significant": False, "significant_streak": 0,
            "perplexity_trend": "not_yet_measured", "can_sample": False,
            "learning_curve": list(previous.get("learning_curve", [])), "samples": []}


def train_and_evaluate(raw_texts: dict[str, str], previous: dict | None = None,
                       train_seconds: float = DEFAULT_TRAIN_SECONDS,
                       max_steps: int | None = None) -> dict:
    previous = previous or {}
    texts = {u: n for u, t in raw_texts.items() if len(n := normalise(t)) >= 40}
    train_urls, held_urls = _split(texts, previous)
    train_chars = sum(len(texts[u]) for u in train_urls)
    held_chars = sum(len(texts[u]) for u in held_urls)
    if train_chars < MIN_TRAIN_CHARS or held_chars < MIN_EVAL_CHARS:
        return _insufficient(previous, train_chars, held_chars)

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

    significant_now = (n_sources >= MIN_EVAL_SOURCES and z >= SIGNIFICANCE_Z
                       and evaluated >= MIN_EVAL_CHARS)
    prior_sig = previous.get("beats_char_baseline_significant", False)
    confirmed = significant_now and (prior_sig or previous.get("significant_streak", 0) >= 1)
    streak = previous.get("significant_streak", 0) + 1 if significant_now else 0

    final_state = model.state()
    fp = model_fingerprint(final_state, steps_trained)

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
        "status": "beats_char_baseline" if confirmed else
                  "improvement_not_yet_significant" if mean_gain > 0 else "below_char_baseline",
        "benchmark": {"locked": True,
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
        "perplexity_trend": trend, "can_sample": True,
        "beats_char_baseline": confirmed,
        "beats_char_baseline_significant": significant_now,
        "learning_curve": curve,
        "samples": [generate(model, "むかしむかし", 90),
                    generate(model, "おじいさんは", 90)],
        "model_fingerprint": fp,
        "state": {**final_state, "version": VERSION, "steps_trained": steps_trained,
                  "model_fingerprint": fp},
        "limitations": ["character-level, no word or concept supervision",
                        "credit only when the per-source improvement clears a strict "
                        "one-sided z on two consecutive cycles"],
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
