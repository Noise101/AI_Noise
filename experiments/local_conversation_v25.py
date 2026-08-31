"""Bounded conversation practice with a local model as partner, never judge."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass


@dataclass
class PracticeTurn:
    seed: str
    model: str
    noise_utterance: str
    partner_reply: str
    partner_question: str
    observed_forms: list[str]
    noise_followup: str = ""
    practice_metrics: dict | None = None
    evidence_score: float = 0.0
    verified: bool = False
    purpose: str = "conversation practice only"


class OllamaConversationPartner:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model or os.environ.get("AI_NOISE_LOCAL_MODEL", "qwen3:4b")

    def reply(self, utterance: str) -> dict | None:
        schema = {"type": "object", "properties": {
            "reply": {"type": "string"}, "question": {"type": "string"}},
            "required": ["reply", "question"]}
        prompt = (
            "You are a patient conversation partner for a beginning learner. "
            "Reply with at most two short child-level sentences, then ask exactly one short question. "
            "Do not grade the learner and do not claim your reply is authoritative evidence.\n"
            f"Learner: {utterance[:600]}"
        )
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                              "think": False, "format": schema,
                              "options": {"temperature": 0.2, "num_predict": 100}}).encode()
        request = urllib.request.Request(f"{self.base_url}/api/generate", data=payload,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                result = json.load(response)
            parsed = json.loads(result.get("response", "{}"))
            reply, question = str(parsed.get("reply", "")).strip(), str(parsed.get("question", "")).strip()
            return {"reply": reply[:500], "question": question[:240]} if reply and question else None
        except Exception:
            return None


def make_noise_utterance(seed: str, mastery: dict, curiosity: dict[str, dict]) -> str:
    goal = mastery.get("next_mastery_goal", {})
    wanting = [(item.get("pressure", 0), gap_id) for gap_id, item in curiosity.items()
               if item.get("status") == "wanting_to_know"]
    unknown = max(wanting, default=(0, "something"))[1].split(":", 1)[-1]
    dimension = goal.get("dimension", "language")
    return (f"I am learning about {seed}. I want to improve my {dimension}. "
            f"I have seen '{unknown}' many times, but I do not understand it yet. "
            f"Can we talk about it with a simple example?")


def practice_once(seed: str, mastery: dict, curiosity: dict[str, dict], partner=None) -> dict:
    partner = partner or OllamaConversationPartner()
    utterance = make_noise_utterance(seed, mastery, curiosity)
    response = partner.reply(utterance)
    if not response:
        return {"status": "local_partner_unavailable", "seed": seed, "noise_utterance": utterance,
                "evidence_score": 0.0, "verified": False}
    observed = sorted(set(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", response["reply"].lower())))
    question_words = [word.lower() for word in re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?", response["question"])]
    known = [word for word in seed.split() if word.isalpha()]
    focus = next((word for word in question_words if word in observed or word in known),
                 known[0] if known else "this")
    followup = f"I think about {focus}. I am not certain yet. I want another example."
    followup_words = set(re.findall(r"[a-z]+", followup.lower()))
    relevant = set(question_words) | set(known) | set(observed)
    metrics = {
        "formed_followup": True,
        "relevant_token_overlap": round(len(followup_words & relevant) / max(1, len(followup_words)), 3),
        "admits_uncertainty": "not certain" in followup.lower(),
        "independent_evidence_added": False,
    }
    turn = PracticeTurn(seed, getattr(partner, "model", "local"), utterance,
                        response["reply"], response["question"], observed, followup, metrics)
    return {"status": "practiced", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **asdict(turn)}
