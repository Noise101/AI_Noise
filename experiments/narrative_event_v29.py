#!/usr/bin/env python3
"""Transparent narrative-event extraction with explicit quality rejection."""

from __future__ import annotations

import re
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
    "grows", "shines", "sings",
}
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

    def extract(self, sentence: str) -> EventExtraction:
        words = [word.lower() for word in WORD.findall(sentence)]
        if len(words) < 3:
            return EventExtraction(sentence, False, None, "too_short", 0.0)
        metadata_hits = set(words) & METADATA_TERMS
        if metadata_hits:
            return EventExtraction(sentence, False, None,
                                   "metadata:" + ",".join(sorted(metadata_hits)), 0.0)
        if sentence.strip().endswith(":"):
            return EventExtraction(sentence, False, None, "heading", 0.0)

        verb_index = next((i for i, word in enumerate(words[1:], 1)
                           if word in VERBS or word in AUXILIARIES), None)
        if verb_index is None:
            verb_index = next((i for i, word in enumerate(words[2:], 2)
                               if word.endswith(("ed", "ing"))), None)
        if verb_index is None:
            return EventExtraction(sentence, False, None, "no_explicit_action", 0.0)

        action_index = verb_index
        if words[verb_index] in AUXILIARIES:
            candidate = next((i for i in range(verb_index + 1, min(len(words), verb_index + 4))
                              if words[i] in VERBS or words[i].endswith(("ed", "ing"))), None)
            if candidate is None:
                return EventExtraction(sentence, False, None, "auxiliary_without_action", 0.0,
                                       verb_index)
            action_index = candidate

        subject_candidates = [word for word in words[:verb_index]
                              if word not in ARTICLES | POSSESSIVES]
        if not subject_candidates:
            return EventExtraction(sentence, False, None, "missing_subject", 0.0, verb_index)
        subject = subject_candidates[-1]
        if subject in PRONOUNS:
            return EventExtraction(sentence, False, None, "unresolved_pronoun_subject", 0.0,
                                   verb_index)
        object_words = [word for word in words[action_index + 1:] if word not in ARTICLES]
        event = Event(subject, words[action_index], "_".join(object_words[:8]))
        quality = 1.0 if object_words else 0.8
        return EventExtraction(sentence, True, event, "accepted", quality, verb_index)
