#!/usr/bin/env python3
"""Say something and learn whether it was understood.

Template utterances cannot get better -- there is no "say it badly and be
corrected" signal.  This module has Noise *compose* an utterance (from observed
events, and increasingly from the character RNN once it can generate), send it
to the local model as an *environment* (not a teacher), and score itself on
whether the reply shows comprehension:

  * the reply shares content words / entities with the utterance
  * the reply is not a clarification request
  * the reply carries the same entity forward

Reward is communicative success, measured behaviourally.  Noise learns which
composition strategy gets understood more often.  The partner's reply is
evidence score zero for *facts*; "was I understood" is a signal about Noise's
own output, not a fact claim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

from coreference_v1 import ANIMATE_HINTS
from narrative_event_v29 import VERBS

VERSION = 1
STRATEGIES = ("event", "event_pair", "rnn_extend", "rnn_free")
# Explicit non-comprehension phrases only.  A bare "?" is not one of them: a
# partner who asks a follow-up question ("Did it win a race?") has usually
# understood and is engaging with the topic.
CLARIFICATION_MARKERS = (
    "what do you mean", "i don't understand", "i do not understand", "unclear",
    "could you clarify", "can you clarify", "what are you trying", "not sure what",
    "don't quite understand", "can't tell what you mean", "cannot tell what you mean",
    "can't quite tell", "doesn't make sense", "does not make sense", "confus",
    "please rephrase", "i can't tell what", "i cannot tell what")
_WORD = re.compile(r"[A-Za-z]+")
_STOP = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "is",
         "was", "were", "be", "it", "he", "she", "they", "his", "her", "with",
         "for", "that", "this", "i", "you", "we", "not", "as", "so", "then"}


class DialoguePartner:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model or os.environ.get("AI_NOISE_LOCAL_MODEL", "qwen3.8:27b")

    def available(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(request, timeout=2) as response:
                return any(item.get("name") == self.model
                           for item in json.load(response).get("models", []))
        except Exception:
            return False

    def respond(self, utterance: str) -> str | None:
        prompt = ("A language learner said the sentence below. Respond naturally in one "
                  "or two short sentences, continuing the topic. If you cannot tell what "
                  "they mean, say so plainly. Do not correct their grammar.\n"
                  f"Learner: {utterance[:400]}")
        schema = {"type": "object", "properties": {"reply": {"type": "string"}},
                  "required": ["reply"]}
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                              "think": False, "format": schema,
                              "options": {"temperature": 0.4, "num_predict": 70}}).encode()
        request = urllib.request.Request(f"{self.base_url}/api/generate", data=payload,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                parsed = json.loads(json.load(response).get("response", "{}"))
            return str(parsed.get("reply", "")).strip()[:400] or None
        except Exception:
            return None


def _content(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in _STOP and len(w) > 2}


_PARTICLES = {"up", "out", "in", "on", "off", "down", "back", "away", "over",
              "here", "there", "him", "her", "them", "it", "us", "me"}


def _verbalise(event: str) -> str:
    parts = event.split("|", 2)
    if len(parts) < 2:
        return ""
    subject, verb = parts[0], parts[1]
    obj = parts[2].replace("_", " ") if len(parts) > 2 else ""
    if not obj:
        return f"the {subject} {verb}"
    if obj.split()[0] in _PARTICLES:
        return f"the {subject} {verb} {obj}"
    return f"the {subject} {verb} the {obj}"


def _speakable(event: str) -> bool:
    """A good utterance seed: animate/common subject, a known verb."""
    parts = event.split("|", 2)
    if len(parts) < 2:
        return False
    subject, verb = parts[0], parts[1]
    animate = subject in ANIMATE_HINTS or (
        subject.endswith("s") and subject[:-1] in ANIMATE_HINTS)
    return animate and verb in VERBS


def compose_utterance(strategy: str, events: list[str], rnn_sampler, topic: str | None,
                      offset: int = 0) -> str:
    good = [e for e in events if _speakable(e)] or [e for e in events if e.count("|") >= 2]
    if not good:
        return "the animal moved"
    pick = good[offset % len(good)]
    if strategy == "event_pair":
        subject = pick.split("|", 1)[0]
        pair = [e for e in good if e.split("|", 1)[0] == subject][:2]
        base = ". ".join(_verbalise(e) for e in pair) if len(pair) == 2 else _verbalise(pick)
    else:
        base = _verbalise(pick)
    if strategy == "rnn_free" and rnn_sampler is not None:
        return rnn_sampler((topic or "the ") + " ").strip() or "the animal moved"
    if strategy == "rnn_extend" and rnn_sampler is not None and base:
        return (base + " and " + rnn_sampler(base + " and ").strip()).strip()
    return base or "the animal moved"


def score_comprehension(utterance: str, reply: str) -> dict:
    if not reply:
        return {"understood": False, "shared_ratio": 0.0, "clarification_request": True,
                "entity_continuity": False}
    lower = reply.lower()
    clarification = any(marker in lower for marker in CLARIFICATION_MARKERS)
    said, heard = _content(utterance), _content(reply)
    shared_ratio = len(said & heard) / len(said) if said else 0.0
    entity_continuity = bool(said & heard) and not clarification
    understood = shared_ratio >= 0.34 and not clarification
    return {"understood": understood, "shared_ratio": round(shared_ratio, 3),
            "clarification_request": clarification, "entity_continuity": entity_continuity}


def _beta_rate(ok: int, total: int) -> float:
    return round((ok + 1) / (total + 2), 4)


def run(events: list[str], partner: DialoguePartner, previous: dict | None = None,
        rnn_sampler=None, topic: str | None = None) -> dict:
    previous = previous or {}
    performance = {k: dict(v) for k, v in previous.get("strategy_performance", {}).items()}
    if not partner.available():
        return {**previous, "version": VERSION, "status": "partner_unavailable",
                "strategy_performance": performance, "turns": []}

    rotation = (previous.get("rotation", 0)) % len(STRATEGIES)
    strategy = STRATEGIES[rotation]
    if strategy.startswith("rnn") and rnn_sampler is None:
        strategy = "event"

    utterance = compose_utterance(strategy, events, rnn_sampler, topic,
                                  offset=previous.get("rotation", 0))
    reply = partner.respond(utterance)
    score = score_comprehension(utterance, reply or "")

    bucket = performance.setdefault(strategy, {"turns": 0, "understood": 0})
    bucket["turns"] += 1
    bucket["understood"] += int(score["understood"])
    bucket["comprehension_rate"] = _beta_rate(bucket["understood"], bucket["turns"])

    best_strategy = max(
        (k for k, v in performance.items() if v["turns"] >= 3),
        key=lambda k: performance[k]["comprehension_rate"], default=None)

    total_turns = sum(v["turns"] for v in performance.values())
    total_understood = sum(v["understood"] for v in performance.values())
    history = list(previous.get("comprehension_history", []))
    history.append({"total_turns": total_turns,
                    "comprehension_rate": _beta_rate(total_understood, total_turns)})
    history = history[-200:]
    tail = [p["comprehension_rate"] for p in history[-8:]]
    trend = "insufficient_data"
    if len(tail) >= 4:
        older = sum(tail[:len(tail) // 2]) / (len(tail) // 2)
        newer = sum(tail[len(tail) // 2:]) / (len(tail) - len(tail) // 2)
        trend = ("improving" if newer > older + 0.02 else
                 "declining" if newer < older - 0.02 else "flat")

    turns = list(previous.get("turns", []))
    turns.append({"strategy": strategy, "utterance": utterance,
                  "partner_reply": reply, **score})
    return {
        "version": VERSION,
        "status": "ran",
        "model": partner.model,
        "rotation": rotation + 1,
        "strategy_performance": performance,
        "best_strategy": best_strategy,
        "overall_comprehension_rate": _beta_rate(total_understood, total_turns),
        "comprehension_history": history,
        "comprehension_trend": trend,
        "turns": turns[-60:],
        "note": "partner replies are evidence score zero for facts; only "
                "'was I understood' is used, as a signal about Noise's own output",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path,
                        default=Path(__file__).resolve().parent.parent / ".local")
    args = parser.parse_args()
    verified = json.loads((args.runtime / "verified-experience.json").read_text(encoding="utf-8"))
    events = [e for seq in verified.get("sequences", []) for e in seq.get("events", [])][:400]
    sampler = None
    seq_path = args.runtime / "sequence-model.json"
    if seq_path.exists():
        from sequence_model_v1 import TinyRNN
        state = json.loads(seq_path.read_text(encoding="utf-8")).get("state")
        if state and state.get("vocab"):
            model = TinyRNN(state["vocab"], state)
            sampler = lambda prime: model.sample(prime, 60, 0.7)
    out = args.runtime / "generative-dialogue.json"
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    report = run(events, DialoguePartner(), previous, sampler)
    out.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in
                      ("status", "overall_comprehension_rate", "comprehension_trend",
                       "best_strategy")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
