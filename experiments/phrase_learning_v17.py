#!/usr/bin/env python3
"""Investigate repeated phrases as possible meaning units, without an LLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from developmental_language_v15 import MultiLevelLearningAgent
from lexical_research_v16 import (
    DefinitionSource, LexicalMeaningLedger, SenseEvidence, WiktionaryDefinitions,
)


class PhraseResearchAgent:
    SENSE_MARKERS = {
        "eventually_after_delay": ("after a long time", "eventually", "finally", "in the end"),
        "repeated_many_times": ("repeatedly", "many times", "over and over", "time after time"),
        "immediately": ("at once", "immediately", "without delay"),
    }

    def __init__(self, sources: list[DefinitionSource]):
        self.sources = sources
        self.ledger = LexicalMeaningLedger()

    @classmethod
    def senses_from_definition(cls, definition: str) -> set[str]:
        lowered = definition.lower()
        return {sense for sense, markers in cls.SENSE_MARKERS.items()
                if any(marker in lowered for marker in markers)}

    @staticmethod
    def compositionality(phrase: str, phrase_belief: dict, word_meanings: dict[str, dict]) -> str:
        parts = phrase.split()
        component_senses = {word_meanings.get(part, {}).get("accepted_sense") for part in parts}
        component_senses.discard(None)
        phrase_sense = phrase_belief.get("accepted_sense")
        if len(component_senses) < len(set(parts)):
            return "unknown_until_component_words_are_grounded"
        if phrase_sense and phrase_sense not in component_senses:
            return "noncompositional_candidate"
        return "possibly_compositional"

    def investigate(self, gap: dict, word_meanings: dict[str, dict]) -> dict:
        phrase = gap["form"]
        documents = []
        for source in self.sources:
            document = source.lookup(phrase)
            if not document:
                continue
            documents.append(document.evidence())
            for definition in document.definitions:
                for sense in self.senses_from_definition(definition):
                    self.ledger.add(SenseEvidence(
                        sense, document.source, document.url, "phrase_definition", document.source_score,
                        hashlib.sha256(definition.encode()).hexdigest(),
                    ))
        belief = self.ledger.belief()
        return {
            "detected_gap": gap, "executed_query": gap["query"], "phrase": phrase,
            "definition_sources": documents, "meaning_belief": belief,
            "compositionality": self.compositionality(phrase, belief, word_meanings),
            "belief_revisions": self.ledger.revisions,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def run(seed_concept: str, max_phrase_investigations: int = 3) -> dict:
    developmental = MultiLevelLearningAgent()
    first = developmental.learn_from_web(seed_concept)
    sources = [
        WiktionaryDefinitions("en.wiktionary.org", "English Wiktionary", 0.82, True),
        WiktionaryDefinitions("simple.wiktionary.org", "Simple English Wiktionary", 0.78, False),
    ]
    investigations = []
    for _ in range(max_phrase_investigations):
        gap = developmental.lexicon.phrase_gap()
        if not gap:
            break
        research = PhraseResearchAgent(sources).investigate(gap, developmental.lexicon.meaning_hypotheses)
        developmental.lexicon.update_phrase_hypothesis(
            gap["form"], research["meaning_belief"], research["compositionality"])
        investigations.append(research)
        if research["meaning_belief"].get("accepted_sense"):
            break
    return {"developmental_learning": first, "phrase_research": investigations,
            "updated_lexicon": developmental.lexicon.report()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("concept", nargs="?", default="fox grapes")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-phrases", type=int, default=3)
    args = parser.parse_args()
    report = run(args.concept, args.max_phrases)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
