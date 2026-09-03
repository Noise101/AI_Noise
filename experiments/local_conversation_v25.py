"""Bounded conversation practice with a local model as partner, never judge."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from narrative_event_v29 import NarrativeEventExtractor, VERBS


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
    expression_type: str = ""
    structural_hypothesis: dict | None = None
    example_comparison: dict | None = None


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
            "Continue this beginner conversation. Give one short contrasting sentence that changes "
            f"only one participant or object and contains the exact expression '{unknown}'. "
            "Keep every other part as similar as possible. Do not repeat the first example. "
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
    question = {
        "relation_phrase": "I want to test what relationship it marks between the things around it.",
        "event_connector": "I want to test how it connects two events.",
        "event_phrase": "I want to test who acts and what participant or object follows it.",
        "noun_phrase": "I want to test what kind of entity the phrase refers to.",
    }.get(expression_type(unknown), "I want to discover its usage from contrasting examples.")
    return (f"I am learning about {seed}. I want to improve my {dimension}. "
            f"I have seen '{unknown}' many times. My current hypothesis is that it links "
            f"nearby ideas, but I am not certain. {question} Please give one simple example and ask me "
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


PREPOSITIONS = {"at", "by", "for", "from", "in", "into", "of", "on", "over", "to", "under", "upon", "with"}
CONNECTORS = {"and", "although", "because", "but", "if", "or", "when", "while"}


def expression_type(expression: str) -> str:
    words = re.findall(r"[a-z]+", expression.lower())
    if not words:
        return "unknown"
    if words[0] in PREPOSITIONS:
        return "relation_phrase"
    if words[0] in CONNECTORS:
        return "event_connector"
    if any(word in VERBS for word in words):
        return "event_phrase"
    if words[0] in {"a", "an", "the", "this", "that", "his", "her", "their"}:
        return "noun_phrase"
    return "lexical_phrase"


def _first_event(text: str) -> dict | None:
    extractor = NarrativeEventExtractor("baseline")
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        result = extractor.extract(sentence)
        if result.accepted and result.event:
            return {"actor": result.event.subject, "action": result.event.action,
                    "object": result.event.object, "sentence": sentence.strip()}
    return None


def compare_examples(expression: str, first: str, second: str) -> tuple[dict, dict]:
    """Create a transparent structural comparison; never ask the model what is true."""
    kind, left, right = expression_type(expression), _first_event(first), _first_event(second)
    comparison = {"first_event": left, "second_event": right,
                  "common_fields": [], "changed_fields": [], "parse_status": "insufficient_structure"}
    if left and right:
        comparison["parse_status"] = "compared"
        for field in ("actor", "action", "object"):
            comparison["common_fields" if left[field] == right[field] else "changed_fields"].append(field)
    words = re.findall(r"[a-z]+", expression.lower())
    if kind == "relation_phrase":
        role = "relation_between_neighboring_entities"
    elif kind == "event_connector":
        role = "connection_between_two_events"
    elif kind == "event_phrase":
        role = "action_followed_by_participant_or_object"
    elif kind == "noun_phrase":
        role = "reference_to_an_entity"
    else:
        role = "usage_cluster_not_yet_identified"
    hypothesis = {"expression": expression, "expression_type": kind, "predicted_role": role,
                  "anchor": words[0] if words else "", "status": "testable_candidate",
                  "evidence_credit": 0,
                  "needs": "independent observed sentences with the same structural role"}
    if not left or not right:
        hypothesis["status"] = "insufficient_structure"
    return comparison, hypothesis


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
    followup = (f"I will compare the actor, action, and object around '{unknown}'. "
                "I am not certain. Please change only one participant or object, keep the exact "
                "expression, and ask what changed.")
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
    comparison, hypothesis = compare_examples(unknown, response["reply"], second["reply"])
    common = ", ".join(comparison["common_fields"]) or "no reliably parsed field"
    changed = ", ".join(comparison["changed_fields"]) or "no single parsed field"
    revision = (f"For '{unknown}', the examples keep {common} and change {changed}. "
                f"My testable candidate is {hypothesis['predicted_role']}. This is still unverified; "
                "I need independent observed sentences to test and possibly reject it.")
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
        "answered_partner_question": True,
        "requested_contrast": bool(second["reply"]),
        "formed_revision": True,
        "awaiting_independent_verification": True,
        "independent_evidence_added": False,
    }
    turn = PracticeTurn(seed, getattr(partner, "model", "local"), utterance,
                        response["reply"], response["question"], observed, followup,
                        second["reply"], second["question"], revision,
                        "hypothesis_example_revision", metrics, 0.0, False,
                        "conversation practice only", unknown, hypothesis["predicted_role"],
                        hypothesis["expression_type"], hypothesis, comparison)
    return {"status": "practiced", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **asdict(turn)}
