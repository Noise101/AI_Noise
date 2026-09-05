#!/usr/bin/env python3
"""Transparent narrative-event extraction with explicit quality rejection."""

from __future__ import annotations

import re
import os
from dataclasses import asdict, dataclass

from story_learning_v12 import Event


WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
ARTICLES = {"a", "an", "the"}
POSSESSIVES = {"her", "his", "their", "its"}
PRONOUNS = {"he", "she", "it", "they", "we", "i", "you", "this", "that"}
AUXILIARIES = {"am", "are", "is", "was", "were", "be", "been", "being",
               "have", "has", "had", "do", "does", "did", "will", "would",
               "could", "should", "can", "may", "might", "must"}
VERBS = {
    "say", "said", "see", "saw", "seen", "go", "went", "come", "came", "get", "got",
    "make", "made", "take", "took", "give", "gave", "find", "found", "know", "knew",
    "think", "thought", "turn", "turned", "eat", "ate", "drink", "drank", "jump",
    "jumped", "run", "ran", "fall", "fell", "grow", "grew", "want", "wanted", "try",
    "tried", "reach", "reached", "miss", "missed", "wait", "waited", "ask", "asked",
    "answer", "answered", "leave", "left", "learn", "learnt", "refuse", "refused",
    "place", "placed", "hang", "hung", "resort", "resorted", "push", "pushed", "sing",
    "sang", "shine", "shone", "sees", "jumps", "waits", "pushes", "eats", "falls",
    "grows", "shines", "sings", "fly", "flew", "flies",
    "become", "became", "change", "changed",
}
SUBJECT_STOP = {"at", "by", "for", "from", "in", "into", "of", "on", "over", "to", "under",
                "upon", "with", "and", "but", "or"}
FUNCTION_WORDS = SUBJECT_STOP | {"as", "during", "before", "after", "while", "than", "then",
                                 "who", "which", "what", "when", "where", "why", "how", "so",
                                 "again", "there", "thus", "if", "unless", "although", "because",
                                 "not", "perhaps", "immediately", "eagerly", "once", "none",
                                 "let", "good", "known"}
METADATA_TERMS = {
    "author", "translator", "translated", "illustrated", "illustrator", "editor", "edition",
    "ebook", "copyright", "license", "gutenberg", "wikisource", "proofread", "transcription",
    "published", "publisher", "language", "contents", "chapter", "volume", "index",
}


@dataclass(frozen=True)
class EventExtraction:
    sentence: str
    accepted: bool
    event: Event | None
    reason: str
    quality: float
    verb_index: int | None = None

    def record(self) -> dict:
        value = asdict(self)
        value["event"] = self.event.key if self.event else None
        return value


class NarrativeEventExtractor:
    """Rule-based baseline: conservative, inspectable, and independent of an LLM."""

    def __init__(self, policy: str | None = None):
        self.policy = policy or os.environ.get("AI_NOISE_PARSER_POLICY", "baseline")

    def extract(self, sentence: str, recent_subject: str | None = None,
                recent_object: str | None = None) -> EventExtraction:
        words = [word.lower() for word in WORD.findall(sentence)]
        if len(words) < 3:
            return EventExtraction(sentence, False, None, "too_short", 0.0)
        metadata_hits = set(words) & METADATA_TERMS
        if metadata_hits:
            return EventExtraction(sentence, False, None,
                                   "metadata:" + ",".join(sorted(metadata_hits)), 0.0)
        if sentence.strip().endswith(":"):
            return EventExtraction(sentence, False, None, "heading", 0.0)

        strict = self.policy.startswith("developmental_grounded")
        try:
            developmental_limit = int(self.policy.rsplit("_", 1)[-1])
        except ValueError:
            developmental_limit = 18
        if strict and (len(words) > developmental_limit
                       or any(mark in sentence for mark in (";", "—", "--"))):
            return EventExtraction(sentence, False, None, "outside_simple_clause", 0.0)

        verb_index = next((i for i, word in enumerate(words[1:], 1)
                           if word in VERBS or word in AUXILIARIES), None)
        if verb_index is None and not strict:
            verb_index = next((i for i, word in enumerate(words[2:], 2)
                               if word.endswith(("ed", "ing"))), None)
        if verb_index is None:
            return EventExtraction(sentence, False, None, "no_explicit_action", 0.0)

        action_index = verb_index
        if words[verb_index] in AUXILIARIES:
            candidate = next((i for i in range(verb_index + 1, min(len(words), verb_index + 4))
                              if words[i] in VERBS or (not strict and
                                 words[i].endswith(("ed", "ing")))), None)
            if candidate is None:
                return EventExtraction(sentence, False, None, "auxiliary_without_action", 0.0,
                                       verb_index)
            action_index = candidate

        excluded_subjects = (ARTICLES | POSSESSIVES | FUNCTION_WORDS if strict
                             else ARTICLES | POSSESSIVES)
        pre_verb = words[:verb_index]
        if strict:
            # A noun inside a prepositional phrase (or a second conjoined noun) that
            # sits between the subject and the verb must not be promoted to subject.
            cut = next((i for i, word in enumerate(pre_verb) if word in SUBJECT_STOP), None)
            if cut:
                pre_verb = pre_verb[:cut]
        subject_candidates = [word for word in pre_verb if word not in excluded_subjects]
        if not subject_candidates:
            return EventExtraction(sentence, False, None, "missing_subject", 0.0, verb_index)
        if self.policy in {"clause_head", "compact_roles"}:
            subject = subject_candidates[0]
        elif strict:
            # The head noun follows any leading adjectives; skip trailing "-ly" adverbs
            # so "The hungry fox saw grapes." yields "fox", not "hungry".
            subject = next((word for word in reversed(subject_candidates)
                            if not word.endswith("ly")), subject_candidates[-1])
        else:
            subject = subject_candidates[-1]
        if subject in SUBJECT_STOP:
            return EventExtraction(sentence, False, None, "invalid_structural_subject", 0.0,
                                   verb_index)
        if strict and (len(subject) < 2 or subject in {"nce"}):
            return EventExtraction(sentence, False, None, "invalid_developmental_subject", 0.0,
                                   verb_index)
        if subject in PRONOUNS:
            if not recent_subject:
                return EventExtraction(sentence, False, None, "unresolved_pronoun_subject", 0.0,
                                       verb_index)
            subject = recent_subject
        object_words = [word for word in words[action_index + 1:]
                        if word not in ARTICLES | (AUXILIARIES if strict else set())]
        if object_words and object_words[-1] in {"it", "them", "him", "her"} and recent_object:
            object_words[-1] = recent_object
        if self.policy in {"compact_roles", "nearest_compact"} or strict:
            compact = next((word for word in object_words
                            if word not in FUNCTION_WORDS | POSSESSIVES | PRONOUNS), "")
            object_value = compact
        else:
            object_value = "_".join(object_words[:8])
        event = Event(subject, words[action_index], object_value)
        quality = (0.95 if strict and object_value else
                   0.85 if subject_candidates[-1] in PRONOUNS else
                   (1.0 if object_words else 0.8))
        reason = "accepted_with_local_coreference" if subject_candidates[-1] in PRONOUNS else "accepted"
        return EventExtraction(sentence, True, event, reason, quality, verb_index)

    def extract_sequence(self, sentences: list[str]) -> list[EventExtraction]:
        results = []
        recent_subject = recent_object = None
        for sentence in sentences:
            result = self.extract(sentence, recent_subject, recent_object)
            results.append(result)
            if result.accepted and result.event:
                recent_subject = result.event.subject
                object_tokens = result.event.object.split("_")
                recent_object = next((token for token in reversed(object_tokens)
                                      if token not in {"him", "her", "it", "them"}), recent_object)
            elif result.reason.startswith("metadata") or result.reason == "heading":
                recent_subject = recent_object = None
        return results

    def extract_multiple(self, sentence: str, recent_subject: str | None = None,
                         recent_object: str | None = None) -> list[EventExtraction]:
        """Split explicit action clauses, retaining the original result when splitting is unsafe."""
        clauses = [part.strip(" ,;:-") for part in re.split(r"\s*;\s*|\s*,\s*|\s+and\s+", sentence,
                                                            flags=re.IGNORECASE)
                   if part.strip(" ,;:-")]
        if len(clauses) < 2:
            return [self.extract(sentence, recent_subject, recent_object)]
        results = []
        subject, obj = recent_subject, recent_object
        for clause in clauses:
            result = self.extract(clause, subject, obj)
            if result.accepted and result.event:
                results.append(result)
                subject = result.event.subject
                tokens = result.event.object.split("_")
                obj = next((token for token in reversed(tokens)
                            if token not in {"him", "her", "it", "them"}), obj)
        if len(results) >= 2:
            return results
        return [self.extract(sentence, recent_subject, recent_object)]

    def extract_multi_sequence(self, sentences: list[str]) -> list[EventExtraction]:
        results = []
        recent_subject = recent_object = None
        for sentence in sentences:
            extracted = self.extract_multiple(sentence, recent_subject, recent_object)
            results.extend(extracted)
            for result in extracted:
                if result.accepted and result.event:
                    recent_subject = result.event.subject
                    tokens = result.event.object.split("_")
                    recent_object = next((token for token in reversed(tokens)
                                          if token not in {"him", "her", "it", "them"}),
                                         recent_object)
                elif result.reason.startswith("metadata") or result.reason == "heading":
                    recent_subject = recent_object = None
        return results
