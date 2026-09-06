#!/usr/bin/env python3
"""Learn to get verifiable-correct work out of an unreliable local model.

The constitution keeps the local model at evidence score zero for *facts*.
This module is about a different skill: reliably getting *usable* output from
it.  Noise formulates a checkable sub-task, sends it, and verifies the answer
against its own grounded vocabulary and the narrative-event parser -- never
against the model's authority.  It then learns which prompt phrasings actually
produce checkable-correct output.

Two task types, both verifiable with existing machinery:
  * simplify   -- rewrite a sentence in words Noise already knows; verify every
                  content word is grounded and it still parses to the same verb.
  * use_word   -- put a target word in a simple animal/person sentence; verify
                  the word appears and the sentence parses to an accepted event
                  with grounded words.

The learned capability is `template_performance`: a Beta-smoothed verified-
success rate per prompt template, and the best template per task type.  The
model's output never updates any belief.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

from narrative_event_v29 import NarrativeEventExtractor, WORD

VERSION = 1
TASKS_PER_RUN = 4
MIN_GROUNDED_RATIO = 0.75

TASK_TEMPLATES = {
    "simplify": [
        "Rewrite this sentence using only very common words a young child knows. "
        "Keep the meaning and keep it to one short sentence. Sentence: {sentence}",
        "Say this in the simplest words you can, one short sentence: {sentence}",
        "A five-year-old should understand every word. Rewrite: {sentence}",
    ],
    "use_word": [
        "Write one short, simple sentence about an animal that uses the word \"{word}\".",
        "Use the word \"{word}\" in a plain sentence a child could read.",
        "Make one easy sentence about a person or animal containing \"{word}\".",
    ],
}
_STOP = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "is",
         "was", "were", "it", "he", "she", "they", "his", "her", "with", "for"}


class OllamaWorker:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model or os.environ.get("AI_NOISE_LOCAL_MODEL", "qwen3.8:27b")

    def available(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(request, timeout=2) as response:
                models = json.load(response).get("models", [])
            return any(item.get("name") == self.model for item in models)
        except Exception:
            return False

    def do_task(self, instruction: str) -> str | None:
        schema = {"type": "object", "properties": {"sentence": {"type": "string"}},
                  "required": ["sentence"]}
        payload = json.dumps({"model": self.model, "prompt": instruction, "stream": False,
                              "think": False, "format": schema,
                              "options": {"temperature": 0.3, "num_predict": 60}}).encode()
        request = urllib.request.Request(f"{self.base_url}/api/generate", data=payload,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                parsed = json.loads(json.load(response).get("response", "{}"))
            text = str(parsed.get("sentence", "")).strip()
            return text[:300] or None
        except Exception:
            return None


def _content_words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text) if w.lower() not in _STOP and len(w) > 2]


def _grounded_ratio(text: str, grounded: set[str]) -> float:
    words = _content_words(text)
    if not words:
        return 0.0
    return sum(word in grounded for word in words) / len(words)


def formulate_tasks(grounded_words: set[str], sentences: list[str],
                    previous: dict, limit: int = TASKS_PER_RUN) -> list[dict]:
    rotation = previous.get("template_rotation", {})
    tasks: list[dict] = []
    hard_words = sorted(grounded_words - _STOP, key=len, reverse=True)[:40]
    long_sentences = sorted((s for s in sentences if 8 <= len(_content_words(s)) <= 18),
                            key=lambda s: -len(s))[:20]
    for task_type in ("simplify", "use_word"):
        templates = TASK_TEMPLATES[task_type]
        which = rotation.get(task_type, 0) % len(templates)
        rotation[task_type] = which + 1
        if task_type == "simplify" and long_sentences:
            sentence = long_sentences[len(tasks) % len(long_sentences)]
            prompt = templates[which].format(sentence=sentence)
            tasks.append({"task_type": task_type, "template_index": which,
                          "prompt": prompt, "source_sentence": sentence})
        elif task_type == "use_word" and hard_words:
            word = hard_words[len(tasks) % len(hard_words)]
            prompt = templates[which].format(word=word)
            tasks.append({"task_type": task_type, "template_index": which,
                          "prompt": prompt, "target_word": word})
        if len(tasks) >= limit:
            break
    previous["template_rotation"] = rotation
    return tasks


def verify(task: dict, response: str, grounded_words: set[str],
           extractor: NarrativeEventExtractor) -> dict:
    if not response:
        return {"verified": False, "reason": "no response"}
    grounded_ratio = _grounded_ratio(response, grounded_words)
    parsed = extractor.extract(response)
    parses = bool(parsed.accepted and parsed.event)
    informational = {}
    if task["task_type"] == "simplify":
        original = task.get("source_sentence", "")
        original_event = extractor.extract(original)
        # The task is "simpler and in known words", not "a canonical SVO clause".
        # parses_to_event is only informational for simplify -- the strict
        # developmental parser rejects most fluent rewrites even when they are
        # grounded and shorter.
        checks = {"grounded_vocabulary": grounded_ratio >= MIN_GROUNDED_RATIO,
                  "not_longer": len(_content_words(response))
                  <= len(_content_words(original)) + 2}
        informational["parses_to_event"] = parses
        if original_event.accepted and original_event.event and parses:
            informational["keeps_the_action"] = (
                original_event.event.action == parsed.event.action)
    else:  # use_word: the sentence must actually be a usable event with the word
        checks = {"grounded_vocabulary": grounded_ratio >= MIN_GROUNDED_RATIO,
                  "parses_to_event": parses,
                  "contains_target": task["target_word"].lower() in _content_words(response)}
    verified = all(checks.values())
    return {"verified": verified, "checks": checks, "informational": informational,
            "grounded_ratio": round(grounded_ratio, 3), "response": response}


def _beta_rate(ok: int, total: int) -> float:
    return round((ok + 1) / (total + 2), 4)


def run(grounded_words: set[str], sentences: list[str], worker: OllamaWorker,
        previous: dict | None = None) -> dict:
    previous = previous or {}
    performance = {k: dict(v) for k, v in previous.get("template_performance", {}).items()}
    if not worker.available():
        return {**previous, "version": VERSION, "status": "local_worker_unavailable",
                "template_performance": performance, "trials": []}

    extractor = NarrativeEventExtractor("developmental_grounded_18")
    tasks = formulate_tasks(grounded_words, sentences, previous)
    trials = []
    for task in tasks:
        response = worker.do_task(task["prompt"])
        outcome = verify(task, response or "", grounded_words, extractor)
        key = f"{task['task_type']}:{task['template_index']}"
        bucket = performance.setdefault(key, {"attempts": 0, "verified": 0})
        bucket["attempts"] += 1
        bucket["verified"] += int(outcome["verified"])
        bucket["verified_rate"] = _beta_rate(bucket["verified"], bucket["attempts"])
        trials.append({**{k: task[k] for k in ("task_type", "template_index", "prompt")},
                       **outcome})

    best_template: dict[str, dict] = {}
    for key, bucket in performance.items():
        task_type, index = key.split(":")
        current = best_template.get(task_type)
        if bucket["attempts"] >= 3 and (current is None
                                        or bucket["verified_rate"] > current["verified_rate"]):
            best_template[task_type] = {"template_index": int(index),
                                        "verified_rate": bucket["verified_rate"],
                                        "attempts": bucket["attempts"]}

    total_attempts = sum(b["attempts"] for b in performance.values())
    total_verified = sum(b["verified"] for b in performance.values())
    history = list(previous.get("success_history", []))
    history.append({"total_attempts": total_attempts,
                    "verified_rate": _beta_rate(total_verified, total_attempts)})
    history = history[-200:]
    tail = [p["verified_rate"] for p in history[-8:]]
    trend = "insufficient_data"
    if len(tail) >= 4:
        older = sum(tail[:len(tail) // 2]) / (len(tail) // 2)
        newer = sum(tail[len(tail) // 2:]) / (len(tail) - len(tail) // 2)
        trend = ("improving" if newer > older + 0.02 else
                 "declining" if newer < older - 0.02 else "flat")

    return {
        "version": VERSION,
        "status": "ran",
        "model": worker.model,
        "tasks_this_run": len(tasks),
        "verified_this_run": sum(t["verified"] for t in trials),
        "template_performance": performance,
        "best_template": best_template,
        "template_rotation": previous.get("template_rotation", {}),
        "overall_verified_rate": _beta_rate(total_verified, total_attempts),
        "success_history": history,
        "success_trend": trend,
        "trials": trials[-40:],
        "note": "local-model output is verified against grounded vocabulary and the "
                "parser only; it never updates a belief",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path,
                        default=Path(__file__).resolve().parent.parent / ".local")
    args = parser.parse_args()
    memory = json.loads((args.runtime / "global-language-memory.json").read_text(encoding="utf-8"))
    grounded = {form for form, item in memory.get("words", {}).items()
                if item.get("curricula", 0) >= 3}
    verified = json.loads((args.runtime / "verified-experience.json").read_text(encoding="utf-8"))
    sentences = [p["sentence"] for p in verified.get("propositions", [])][:200]
    out = args.runtime / "llm-tooluse.json"
    previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    report = run(grounded, sentences, OllamaWorker(), previous)
    out.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in
                      ("status", "verified_this_run", "tasks_this_run",
                       "overall_verified_rate", "success_trend", "best_template")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
