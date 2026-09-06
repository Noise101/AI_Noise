#!/usr/bin/env python3
"""A tiny character-level RNN trained from scratch on Noise's own reading.

No pretrained model, no numpy -- weights are plain nested lists, the forward
pass and truncated BPTT are hand-written.  It is deliberately small (hidden
dim 24, ~4k parameters) and time-boxed per call, and it accumulates across
cycles: each call runs a bounded number of SGD steps on the training split and
carries the updated weights forward in `previous`.

Why it exists (see the architecture assessment):
  * held-out bits/char is a *continuous* capability signal -- it moves with a
    few hundred new sentences, unlike a Bonferroni sign test.
  * a `sample()` head is the missing "generate language, badly, and be
    corrected" mechanism that template utterances cannot provide.

It earns capability credit the same way every predictor here does: only when
its held-out bits/char beats the order-0 character baseline on a frozen,
collection-disjoint split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

VERSION = 1
HIDDEN = 24
SEQ_LEN = 48
LEARNING_RATE = 0.02
GRAD_CLIP = 1.0
DEFAULT_TRAIN_SECONDS = 8.0
MIN_TRAIN_CHARS = 4000
MIN_EVAL_CHARS = 2000
MAX_EVAL_CHARS = 25000        # pure-Python eval is O(chars * hidden * vocab)
BENCHMARK_SALT = "sequence-model-benchmark:v1"

_KEEP = re.compile(r"[a-z .,!?'\"-]")


def collection_key(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    path = urllib.parse.unquote(parsed.path).rstrip("/")
    if "/wiki/" in path:
        title = path.split("/wiki/", 1)[1]
        return f"{parsed.netloc}/wiki/{title.rsplit('/', 1)[0] if '/' in title else title}"
    parent = path.rsplit("/", 1)[0] if path.count("/") > 1 else path
    return f"{parsed.netloc}{parent}"


def normalise(text: str) -> str:
    return "".join(_KEEP.findall(text.lower().replace("\n", " ")))


def source_texts(audit_memory: dict) -> dict[str, str]:
    """One normalised text blob per source url, admitted sentences in order."""
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for item in audit_memory.get("records", {}).values():
        if item.get("curriculum_admitted") is True and item.get("source_url") and item.get("sentence"):
            grouped[item["source_url"]].append(
                (item.get("source_position", 0), item["sentence"]))
    texts = {}
    for url, rows in grouped.items():
        blob = " ".join(sentence for _, sentence in sorted(rows))
        clean = normalise(blob)
        if len(clean) >= 40:
            texts[url] = clean
    return texts


# --- vocabulary --------------------------------------------------------------
def build_vocab(texts: dict[str, str]) -> list[str]:
    chars = Counter()
    for text in texts.values():
        chars.update(text)
    vocab = sorted(ch for ch, count in chars.items() if count >= 3)
    return vocab or sorted({ch for text in texts.values() for ch in text})


# --- model -----------------------------------------------------------------
def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def _randn(rows: int, cols: int, scale: float, rng: random.Random) -> list[list[float]]:
    return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


class TinyRNN:
    def __init__(self, vocab: list[str], state: dict | None = None):
        self.vocab = vocab
        self.V = len(vocab)
        self.index = {ch: i for i, ch in enumerate(vocab)}
        if state and state.get("vocab") == vocab and state.get("Whh"):
            self.Wxh = state["Wxh"]
            self.Whh = state["Whh"]
            self.Why = state["Why"]
            self.bh = state["bh"]
            self.by = state["by"]
        else:
            rng = random.Random(1234)
            self.Wxh = _randn(HIDDEN, self.V, 0.1, rng)
            self.Whh = _randn(HIDDEN, HIDDEN, 0.1, rng)
            self.Why = _randn(self.V, HIDDEN, 0.1, rng)
            self.bh = [0.0] * HIDDEN
            self.by = [0.0] * self.V

    def state(self) -> dict:
        return {"vocab": self.vocab, "Wxh": self.Wxh, "Whh": self.Whh,
                "Why": self.Why, "bh": self.bh, "by": self.by}

    def _step(self, x_index: int, h_prev: list[float]) -> tuple[list[float], list[float]]:
        h = [0.0] * HIDDEN
        for i in range(HIDDEN):
            acc = self.bh[i] + self.Wxh[i][x_index]
            whh_i = self.Whh[i]
            for j in range(HIDDEN):
                acc += whh_i[j] * h_prev[j]
            h[i] = math.tanh(acc)
        y = [0.0] * self.V
        for k in range(self.V):
            acc = self.by[k]
            why_k = self.Why[k]
            for i in range(HIDDEN):
                acc += why_k[i] * h[i]
            y[k] = acc
        m = max(y)
        exps = [math.exp(v - m) for v in y]
        total = sum(exps)
        probs = [v / total for v in exps]
        return h, probs

    def bits_per_char(self, text: str) -> tuple[float, int]:
        if len(text) < 2:
            return 0.0, 0
        h = [0.0] * HIDDEN
        total_nll = 0.0
        n = 0
        for a, b in zip(text, text[1:]):
            xi = self.index.get(a)
            yi = self.index.get(b)
            if xi is None or yi is None:
                h = [0.0] * HIDDEN
                continue
            h, probs = self._step(xi, h)
            total_nll += -math.log(max(probs[yi], 1e-12))
            n += 1
        return (total_nll / n / math.log(2)) if n else 0.0, n

    def train_step(self, text: str, lr: float) -> float:
        text = text[:SEQ_LEN + 1]
        indices = [self.index.get(ch) for ch in text]
        if any(i is None for i in indices) or len(indices) < 2:
            return 0.0
        hs = [[0.0] * HIDDEN]
        ps: list[list[float]] = []
        loss = 0.0
        for t in range(len(indices) - 1):
            h, probs = self._step(indices[t], hs[-1])
            hs.append(h)
            ps.append(probs)
            loss += -math.log(max(probs[indices[t + 1]], 1e-12))
        dWxh = _zeros(HIDDEN, self.V)
        dWhh = _zeros(HIDDEN, HIDDEN)
        dWhy = _zeros(self.V, HIDDEN)
        dbh = [0.0] * HIDDEN
        dby = [0.0] * self.V
        dh_next = [0.0] * HIDDEN
        for t in range(len(indices) - 2, -1, -1):
            probs = ps[t]
            target = indices[t + 1]
            dy = list(probs)
            dy[target] -= 1.0
            h = hs[t + 1]
            for k in range(self.V):
                dyk = dy[k]
                if dyk:
                    dby[k] += dyk
                    dWhy_k = dWhy[k]
                    for i in range(HIDDEN):
                        dWhy_k[i] += dyk * h[i]
            dh = list(dh_next)
            for i in range(HIDDEN):
                acc = 0.0
                for k in range(self.V):
                    acc += self.Why[k][i] * dy[k]
                dh[i] += acc
            draw = [dh[i] * (1.0 - h[i] * h[i]) for i in range(HIDDEN)]
            h_prev = hs[t]
            xi = indices[t]
            for i in range(HIDDEN):
                dbh[i] += draw[i]
                dWxh[i][xi] += draw[i]
                dWhh_i = dWhh[i]
                for j in range(HIDDEN):
                    dWhh_i[j] += draw[i] * h_prev[j]
            dh_next = [0.0] * HIDDEN
            for j in range(HIDDEN):
                acc = 0.0
                for i in range(HIDDEN):
                    acc += self.Whh[i][j] * draw[i]
                dh_next[j] = acc

        def clip(value: float) -> float:
            return max(-GRAD_CLIP, min(GRAD_CLIP, value))

        for i in range(HIDDEN):
            self.bh[i] -= lr * clip(dbh[i])
            for x in range(self.V):
                if dWxh[i][x]:
                    self.Wxh[i][x] -= lr * clip(dWxh[i][x])
            for j in range(HIDDEN):
                self.Whh[i][j] -= lr * clip(dWhh[i][j])
        for k in range(self.V):
            self.by[k] -= lr * clip(dby[k])
            for i in range(HIDDEN):
                self.Why[k][i] -= lr * clip(dWhy[k][i])
        return loss / max(1, len(indices) - 1)

    def sample(self, prime: str, length: int = 120, temperature: float = 0.8,
               rng: random.Random | None = None) -> str:
        rng = rng or random.Random()
        h = [0.0] * HIDDEN
        out = []
        prime = normalise(prime) or " "
        last = None
        for ch in prime:
            xi = self.index.get(ch)
            if xi is None:
                continue
            h, probs = self._step(xi, h)
            last = xi
        if last is None:
            last = self.index.get(" ", 0)
        for _ in range(length):
            h, probs = self._step(last, h)
            if temperature != 1.0:
                logits = [math.log(max(p, 1e-12)) / temperature for p in probs]
                m = max(logits)
                exps = [math.exp(v - m) for v in logits]
                total = sum(exps)
                probs = [v / total for v in exps]
            r = rng.random()
            cumulative = 0.0
            choice = len(probs) - 1
            for idx, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    choice = idx
                    break
            out.append(self.vocab[choice])
            last = choice
        return "".join(out)


# --- frozen benchmark + training loop -------------------------------------
def _split(texts: dict[str, str], previous: dict) -> tuple[set[str], set[str]]:
    locked = previous.get("benchmark", {})
    if locked.get("held_out_collections"):
        held = set(locked["held_out_collections"])
        return ({u for u in texts if collection_key(u) not in held},
                {u for u in texts if collection_key(u) in held})
    collections = sorted({collection_key(u) for u in texts},
                         key=lambda c: hashlib.sha256(f"{BENCHMARK_SALT}:{c}".encode()).hexdigest())
    held = set(collections[:max(1, len(collections) // 5)])
    return ({u for u in texts if collection_key(u) not in held},
            {u for u in texts if collection_key(u) in held})


def train_and_evaluate(audit_memory: dict, previous: dict | None = None,
                       train_seconds: float = DEFAULT_TRAIN_SECONDS,
                       max_steps: int | None = None) -> dict:
    previous = previous or {}
    texts = source_texts(audit_memory)
    train_urls, held_urls = _split(texts, previous)
    train_chars = sum(len(texts[u]) for u in train_urls)
    held_chars = sum(len(texts[u]) for u in held_urls)

    if train_chars < MIN_TRAIN_CHARS or held_chars < MIN_EVAL_CHARS:
        return {"version": VERSION, "status": "insufficient_text",
                "benchmark": {"locked": False},
                "train_chars": train_chars, "held_out_chars": held_chars,
                "held_out_bits_per_char": None, "baseline_bits_per_char": None,
                "perplexity_trend": "not_yet_measured", "can_sample": False,
                "learning_curve": list(previous.get("learning_curve", [])), "samples": []}

    vocab = previous.get("state", {}).get("vocab") or build_vocab(texts)
    model = TinyRNN(vocab, previous.get("state"))
    rng = random.Random(previous.get("steps_trained", 0) + 1)

    train_windows: list[str] = []
    for url in sorted(train_urls):
        text = texts[url]
        for start in range(0, max(1, len(text) - SEQ_LEN), SEQ_LEN):
            train_windows.append(text[start:start + SEQ_LEN + 1])
    rng.shuffle(train_windows)

    deadline = time.monotonic() + train_seconds
    steps = 0
    loss_sum = 0.0
    for window in train_windows:
        if (max_steps is not None and steps >= max_steps) or time.monotonic() > deadline:
            break
        loss_sum += model.train_step(window, LEARNING_RATE)
        steps += 1
    steps_trained = previous.get("steps_trained", 0) + steps

    # order-0 baseline from the training char distribution
    base_counts = Counter()
    for url in train_urls:
        base_counts.update(texts[url])
    base_total = sum(base_counts.values())
    base_nll = 0.0
    held_n = 0
    for url in held_urls:
        for ch in texts[url]:
            p = base_counts.get(ch, 0.5) / base_total
            base_nll += -math.log(max(p, 1e-12))
            held_n += 1
    baseline_bpc = (base_nll / held_n / math.log(2)) if held_n else 0.0

    model_bpc, evaluated = 0.0, 0
    for url in sorted(held_urls):
        if evaluated >= MAX_EVAL_CHARS:
            break
        bpc, n = model.bits_per_char(texts[url][:MAX_EVAL_CHARS])
        model_bpc += bpc * n
        evaluated += n
    model_bpc = model_bpc / evaluated if evaluated else 0.0

    learning_curve = list(previous.get("learning_curve", []))
    point = {"steps_trained": steps_trained, "train_chars": train_chars,
             "held_out_bits_per_char": round(model_bpc, 4),
             "baseline_bits_per_char": round(baseline_bpc, 4),
             "improvement_bits": round(baseline_bpc - model_bpc, 4)}
    if not learning_curve or learning_curve[-1]["steps_trained"] != steps_trained:
        learning_curve.append(point)
    learning_curve = learning_curve[-200:]

    tail = [p["held_out_bits_per_char"] for p in learning_curve[-8:]]
    if len(tail) >= 4:
        older = sum(tail[:len(tail) // 2]) / (len(tail) // 2)
        newer = sum(tail[len(tail) // 2:]) / (len(tail) - len(tail) // 2)
        trend = "improving" if newer < older - 0.01 else "declining" if newer > older + 0.01 else "flat"
    else:
        trend = "insufficient_data"

    beats_baseline = model_bpc < baseline_bpc and evaluated >= MIN_EVAL_CHARS
    samples = [model.sample("the ", 100, 0.7, rng), model.sample("a ", 100, 0.9, rng)]

    return {
        "version": VERSION,
        "status": "beats_char_baseline" if beats_baseline else "below_char_baseline",
        "benchmark": {"locked": True,
                      "held_out_collections": sorted({collection_key(u) for u in held_urls}),
                      "train_sources": len(train_urls), "held_out_sources": len(held_urls)},
        "hidden_dim": HIDDEN, "vocab_size": len(vocab),
        "train_chars": train_chars, "held_out_chars": held_chars,
        "steps_this_cycle": steps, "steps_trained": steps_trained,
        "mean_train_loss": round(loss_sum / steps, 4) if steps else None,
        "held_out_bits_per_char": round(model_bpc, 4),
        "baseline_bits_per_char": round(baseline_bpc, 4),
        "improvement_bits": round(baseline_bpc - model_bpc, 4),
        "perplexity_trend": trend,
        "can_sample": True,
        "beats_char_baseline": beats_baseline,
        "learning_curve": learning_curve,
        "samples": samples,
        "state": model.state(),
        "limitations": ["character-level; no word or concept supervision",
                        "credit only when held-out bits/char beats the order-0 baseline"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path,
                        default=Path(__file__).resolve().parent.parent / ".local")
    parser.add_argument("--seconds", type=float, default=DEFAULT_TRAIN_SECONDS)
    args = parser.parse_args()
    audit = json.loads((args.runtime / "parser-audit-memory.json").read_text(encoding="utf-8"))
    out = args.runtime / "sequence-model.json"
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    report = train_and_evaluate(audit, previous, args.seconds)
    out.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("status", "held_out_bits_per_char", "baseline_bits_per_char",
                       "perplexity_trend", "steps_trained", "samples")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
