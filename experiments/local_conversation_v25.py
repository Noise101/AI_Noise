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
    partner_second_reply: str = ""
    partner_second_question: str = ""
    noise_revision: str = ""
    dialogue_stage: str = "hypothesis_example_revision"
    practice_metrics: dict | None = None
    evidence_score: float = 0.0
    verified: bool = False
    purpose: str = "conversation practice only"
    unknown_expression: str = ""
    hypothesis_focus: str = ""


class OllamaConversationPartner:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model or os.environ.get("AI_NOISE_LOCAL_MODEL", "qwen3.8:27b")

    def _request(self, prompt: str) -> dict | None:
        schema = {"type": "object", "properties": {
            "reply": {"type": "string"}, "question": {"type": "string"}},
            "required": ["reply", "question"]}
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

    def reply(self, utterance: str) -> dict | None:
        return self._request(
            "You are a patient conversation partner for a beginning learner. "
            "Reply with at most two short child-level sentences, then ask exactly one short question. "
            "The example must contain the exact expression quoted by the learner. "
            "Do not grade the learner and do not claim your reply is authoritative evidence.\n"
            f"Learner: {utterance[:600]}")

    def contrast(self, unknown: str, transcript: str) -> dict | None:
        return self._request(
            "Continue this beginner conversation. Give one genuinely different or contrasting "
            f"example containing the exact expression '{unknown}'. Do not repeat the first example. "
            "Then ask exactly one short question about what changed. Do not grade the learner and "
            "do not claim authority.\nConversation: " + transcript[:1000])


def select_dialogue_unknown(curiosity: dict[str, dict], verification_memory: dict | None = None) -> str:
    investigated = (verification_memory or {}).get("expressions", {})
    wanting = [(item.get("pressure", 0), gap_id.split(":", 1)[-1])
               for gap_id, item in curiosity.items() if item.get("status") == "wanting_to_know"]
    if not wanting:
        return "something"
    open_questions = [item for item in wanting
                      if investigated.get(item[1], {}).get("independent_sources", 0) < 2]
    if open_questions:
        wanting = open_questions
    # Curiosity still supplies priority, while repeated verified questions lose novelty.
    return max(wanting, key=lambda item: (
        item[0] / (1 + investigated.get(item[1], {}).get("attempts", 0)) ** 2,
        -investigated.get(item[1], {}).get("attempts", 0), item[1]))[1]


def make_noise_utterance(seed: str, mastery: dict, curiosity: dict[str, dict],
                         verification_memory: dict | None = None) -> str:
    goal = mastery.get("next_mastery_goal", {})
    unknown = select_dialogue_unknown(curiosity, verification_memory)
    dimension = goal.get("dimension", "language")
    return (f"I am learning about {seed}. I want to improve my {dimension}. "
            f"I have seen '{unknown}' many times. My current hypothesis is that it links "
            f"nearby ideas, but I am not certain. Please give one simple example and ask me "
            f"what I predict it means there.")


QUESTION_STOP = {"a", "an", "and", "are", "can", "did", "do", "does", "in", "is", "it",
                 "me", "of", "something", "tell", "that", "the", "think", "to", "what",
                 "what's", "where", "which", "who", "why", "with", "you", "your", "okay",
                 "let's", "try", "example", "changed", "compared", "sentence", "different",
                 "means", "meaning", "now", "shows", "this", "together", "contrasting",
                 "contrast", "differ", "note", "using", "both", "examples", "between", "two"}


def _focus_from_question(question: str, unknown: str, seed: str) -> str:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", question.lower())
    preferred = [word for word in words if word not in QUESTION_STOP and len(word) > 2]
    seed_words = [word for word in seed.lower().split() if word.isalpha()]
    return next((word for word in preferred if word not in seed_words),
                preferred[0] if preferred else (unknown or (seed_words[0] if seed_words else "this")))


def practice_once(seed: str, mastery: dict, curiosity: dict[str, dict], partner=None,
                  verification_memory: dict | None = None) -> dict:
    partner = partner or OllamaConversationPartner()
    unknown = select_dialogue_unknown(curiosity, verification_memory)
    utterance = make_noise_utterance(seed, mastery, curiosity, verification_memory)
    response = partner.reply(utterance)
    if not response:
        return {"status": "local_partner_unavailable", "seed": seed, "noise_utterance": utterance,
                "unknown_expression": unknown, "hypothesis_focus": "",
                "practice_metrics": {}, "evidence_score": 0.0, "verified": False}
    focus = _focus_from_question(response["question"], unknown, seed)
    followup = (f"My current answer is that '{unknown}' connects the example to {focus}. "
                "I am not certain. Please show a contrasting example and ask what changed.")
    transcript = (f"Learner asked about '{unknown}'. Partner said: {response['reply']} "
                  f"Partner asked: {response['question']} Learner answered: {followup}")
    second = (partner.contrast(unknown, transcript) if hasattr(partner, "contrast")
              else partner.reply(transcript))
    second = second or {"reply": "", "question": ""}
    first_words = set(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", response["question"].lower()))
    seed_words = set(word for word in seed.lower().split() if word.isalpha())
    second_words = [word for word in re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?", (second["reply"] + " " + second["question"]).lower())
        if word not in QUESTION_STOP and len(word) > 2]
    contrast = next((word for word in second_words
                     if word not in first_words and word not in seed_words),
                    second_words[0] if second_words else focus)
    revision = (f"I now have two possibilities for '{unknown}': it may connect to {focus}, "
                f"or the contrast may involve {contrast}. This is still unverified; I need "
                "an independent observed sentence to test it.")
    observed = sorted(set(re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?", (response["reply"] + " " + second["reply"]).lower())))
    question_words = [word.lower() for word in re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?", response["question"] + " " + second["question"])]
    known = [word for word in seed.split() if word.isalpha()]
    followup_words = set(re.findall(r"[a-z]+", followup.lower()))
    relevant = set(question_words) | set(known) | set(observed)
    metrics = {
        "formed_followup": True,
        "relevant_token_overlap": round(len(followup_words & relevant) / max(1, len(followup_words)), 3),
        "admits_uncertainty": "not certain" in followup.lower(),
        "answered_partner_question": focus not in QUESTION_STOP,
        "requested_contrast": bool(second["reply"]),
        "formed_revision": True,
        "awaiting_independent_verification": True,
        "independent_evidence_added": False,
    }
    turn = PracticeTurn(seed, getattr(partner, "model", "local"), utterance,
                        response["reply"], response["question"], observed, followup,
                        second["reply"], second["question"], revision,
                        "hypothesis_example_revision", metrics, 0.0, False,
                        "conversation practice only", unknown, focus)
    return {"status": "practiced", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **asdict(turn)}
