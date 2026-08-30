#!/usr/bin/env python3
"""Parallel character, word, phrase, meaning, and story learning without an LLM."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from story_concepts_v14 import StoryConceptAgent
from story_learning_v12 import Event, StoryLearner
from story_web_curriculum_v13 import GutenbergStories, StoryCurriculumAgent, WikisourceStories


LATIN_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
JAPANESE_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]+")
FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "for", "from",
    "he", "she", "it", "they", "his", "her", "their", "this", "that", "with", "but",
}


class DevelopmentalLexicon:
    """Learns multiple linguistic levels from the same observations.

    Whitespace words are observations, not predefined meanings. Japanese has no
    assumed word boundaries: repeated 2-4 character chunks remain candidates
    until grounding supplies evidence.
    """

    def __init__(self):
        self.characters: Counter[str] = Counter()
        self.character_links: Counter[tuple[str, str]] = Counter()
        self.words: Counter[str] = Counter()
        self.word_links: Counter[tuple[str, str]] = Counter()
        self.chunk_candidates: Counter[str] = Counter()
        self.roles: dict[str, Counter[str]] = defaultdict(Counter)
        self.contexts: dict[str, Counter[str]] = defaultdict(Counter)
        self.meaning_revisions: list[dict] = []
        self.meaning_hypotheses: dict[str, dict] = {}
        self.sentences_seen = 0

    @staticmethod
    def _characters(text: str) -> list[str]:
        return [char.lower() for char in text if not char.isspace() and not unicodedata.category(char).startswith("P")]

    @staticmethod
    def _words(text: str) -> list[str]:
        return [word.lower() for word in LATIN_WORD.findall(text)]

    def observe(self, sentence: str, event: Event | None = None) -> None:
        self.sentences_seen += 1
        chars = self._characters(sentence)
        self.characters.update(chars)
        self.character_links.update(zip(chars, chars[1:]))

        words = self._words(sentence)
        self.words.update(words)
        self.word_links.update(zip(words, words[1:]))
        for left, word, right in zip(["<START>"] + words, words, words[1:] + ["<END>"]):
            self.contexts[word][f"left:{left}"] += 1
            self.contexts[word][f"right:{right}"] += 1

        for run in JAPANESE_RUN.findall(sentence):
            for size in range(2, min(4, len(run)) + 1):
                self.chunk_candidates.update(run[index:index + size] for index in range(len(run) - size + 1))

        if event:
            self._ground(event.subject, "agent")
            self._ground(event.action, "action")
            for token in event.object.split("_"):
                if token:
                    self._ground(token, "object_or_detail")

    def _ground(self, word: str, role: str) -> None:
        word = word.lower()
        if word in FUNCTION_WORDS:
            return
        before = self.dominant_role(word)
        self.roles[word][role] += 1
        after = self.dominant_role(word)
        if before and before != after:
            self.meaning_revisions.append({
                "word": word, "before": before, "after": after,
                "reason": "new grounded uses changed the dominant semantic role",
            })

    def dominant_role(self, word: str) -> str | None:
        counts = self.roles.get(word)
        if not counts:
            return None
        return max(counts, key=lambda role: (counts[role], role))

    def update_meaning_hypothesis(self, word: str, belief: dict) -> None:
        """Write researched lexical meaning back into long-lived vocabulary memory."""
        word = word.lower()
        before = self.meaning_hypotheses.get(word)
        compact = {
            "status": belief.get("status"), "accepted_sense": belief.get("accepted_sense"),
            "leading_sense": belief.get("leading_sense"),
            "confidence_margin": belief.get("confidence_margin", 0.0),
            "alternatives": belief.get("alternatives", []),
        }
        self.meaning_hypotheses[word] = compact
        if before and before.get("accepted_sense") != compact.get("accepted_sense"):
            self.meaning_revisions.append({
                "word": word, "before": before.get("accepted_sense"),
                "after": compact.get("accepted_sense"),
                "reason": "new sourced lexical evidence revised the stored sense",
            })

    def phrase_candidates(self, minimum_count: int = 2) -> list[dict]:
        phrases = []
        for (left, right), count in self.word_links.items():
            if count < minimum_count:
                continue
            association = count / math.sqrt(self.words[left] * self.words[right])
            phrases.append({"phrase": f"{left} {right}", "count": count,
                            "association": round(association, 3), "kind": "word_phrase_candidate"})
        for chunk, count in self.chunk_candidates.items():
            if count >= minimum_count:
                phrases.append({"phrase": chunk, "count": count,
                                "association": None, "kind": "unsegmented_chunk_candidate"})
        return sorted(phrases, key=lambda item: (-item["count"], item["phrase"]))

    def lexical_gap(self) -> dict | None:
        unknown = [(count, word) for word, count in self.words.items()
                   if word not in FUNCTION_WORDS and not self.roles.get(word)
                   and not self.meaning_hypotheses.get(word, {}).get("accepted_sense")]
        if unknown:
            count, word = max(unknown, key=lambda item: (item[0], item[1]))
            contexts = [name.split(":", 1)[1] for name, _ in self.contexts[word].most_common(4)]
            return {"kind": "unknown_word_meaning", "form": word, "observations": count,
                    "contexts": contexts, "query": f'"{word}" meaning simple story example'}
        chunks = [(count, chunk) for chunk, count in self.chunk_candidates.items() if count >= 2]
        if chunks:
            count, chunk = max(chunks, key=lambda item: (item[0], len(item[1]), item[1]))
            return {"kind": "unknown_word_boundary", "form": chunk, "observations": count,
                    "query": f'"{chunk}" やさしい 文 意味'}
        return None

    def report(self) -> dict:
        grounded = []
        for word, role_counts in sorted(self.roles.items()):
            total = sum(role_counts.values())
            grounded.append({
                "form": word, "roles": dict(role_counts), "dominant_role": self.dominant_role(word),
                "confidence": round(max(role_counts.values()) / total, 3),
            })
        return {
            "sentences_seen": self.sentences_seen,
            "character_inventory": len(self.characters),
            "characters": dict(self.characters.most_common()),
            "word_forms": dict(self.words.most_common()),
            "grounded_meanings": grounded,
            "researched_meanings": self.meaning_hypotheses,
            "phrase_candidates": self.phrase_candidates(),
            "meaning_revisions": self.meaning_revisions,
            "next_lexical_goal": self.lexical_gap(),
        }


class MultiLevelLearningAgent:
    def __init__(self):
        self.lexicon = DevelopmentalLexicon()
        self.story = StoryLearner()
        self.concepts = StoryConceptAgent()

    def observe_source(self, source: str, url: str, score: float, sentences: list[str]) -> None:
        events = [StoryCurriculumAgent.parse_child_event(sentence) for sentence in sentences]
        grounded_events = []
        for sentence, event in zip(sentences, events):
            self.lexicon.observe(sentence, event)
            if event:
                grounded_events.append(event)
        self.story.observe_events(grounded_events)
        self.concepts.ingest(source, url, score, sentences)

    def learn_from_web(self, seed_concept: str) -> dict:
        curriculum = StoryCurriculumAgent([WikisourceStories(), GutenbergStories()])
        search = curriculum.investigate(seed_concept)
        for observation in curriculum.source_observations:
            self.observe_source(observation["source"], observation["url"], observation["source_score"],
                                observation["sentences"])
        return {
            "seed_concept": seed_concept,
            "first_generated_query": search["generated_query"],
            "sources": search["sources_found"],
            "parallel_learning": {
                "characters_words_phrases": self.lexicon.report(),
                "event_predictions": self.story.report(),
                "cross_source_concepts": self.concepts.ledger.report(),
            },
            "next_self_generated_goal": self.lexicon.lexical_gap(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("concept", nargs="?", default="fox grapes")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = MultiLevelLearningAgent().learn_from_web(args.concept)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
