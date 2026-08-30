#!/usr/bin/env python3
"""Cross-source concept integration and belief revision for child-level stories."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from story_web_curriculum_v13 import GutenbergStories, StoryCurriculumAgent, WikisourceStories


@dataclass(frozen=True)
class ConceptEvidence:
    subject: str
    relation: str
    object: str
    polarity: bool
    scope: str
    source: str
    source_url: str
    source_score: float
    observation_hash: str

    @property
    def proposition(self) -> tuple[str, str, str, str]:
        return self.subject, self.relation, self.object, self.scope


class ConceptExtractor:
    """Small explicit semantic vocabulary; every mapping is inspectable."""

    @staticmethod
    def _has(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    def extract(self, sentence: str, source: str, url: str, score: float,
                context_entities: set[str] | None = None) -> list[ConceptEvidence]:
        text = " ".join(re.findall(r"[a-z]+", sentence.lower()))
        digest = hashlib.sha256(sentence.encode()).hexdigest()
        context_entities = context_entities or set()
        words = set(text.split())
        if "fox" in context_entities and words & {"he", "she", "him", "her", "herself", "himself"}:
            text += " fox"
        if "grapes" in context_entities and words & {"them", "bunch", "morsel", "they", "it"}:
            text += " grapes"
        evidence = []

        def add(relation: str, obj: str, polarity: bool = True, scope: str = "narrator_fact") -> None:
            evidence.append(ConceptEvidence("fox", relation, obj, polarity, scope,
                                            source, url, score, digest))

        mentions_grapes = "grape" in text
        if "fox" in text and mentions_grapes and self._has(text, ("saw", "came to", "found")):
            add("encounters", "grapes")
        if mentions_grapes and self._has(text, ("tried", "tricks", "jump", "reach", "get at")):
            add("attempts_to_obtain", "grapes")
        if mentions_grapes and self._has(text, ("could not reach", "missed", "in vain", "no greater success", "had to give up")):
            add("obtains", "grapes", False)
        if mentions_grapes and self._has(text, ("got the grapes", "reached the grapes", "ate the grapes")):
            add("obtains", "grapes", True)
        if mentions_grapes and self._has(text, ("ripe", "ripening")):
            evidence.append(ConceptEvidence("grapes", "quality", "ripe", True, "narrator_fact",
                                            source, url, score, digest))
        if mentions_grapes and "sour" in text:
            evidence.append(ConceptEvidence("grapes", "quality", "sour", True, "fox_belief",
                                            source, url, score, digest))
        if self._has(text, ("despise what", "sour")):
            add("devalues_after_failure", "grapes", True, "interpretation")
        return evidence


class ConceptLedger:
    def __init__(self):
        self.evidence: list[ConceptEvidence] = []
        self.revisions: list[dict] = []
        self._last_status: dict[tuple[str, str, str, str], tuple[str, bool | None]] = {}

    def add(self, item: ConceptEvidence) -> None:
        before = self.belief(item.proposition)
        self.evidence.append(item)
        after = self.belief(item.proposition)
        old = None if before is None else (before["status"], before["accepted_polarity"])
        new = None if after is None else (after["status"], after["accepted_polarity"])
        if old is not None and old != new:
            self.revisions.append({
                "proposition": list(item.proposition), "before": old, "after": new,
                "trigger_source": item.source, "reason": "new independent evidence changed support balance",
            })

    def belief(self, proposition: tuple[str, str, str, str]) -> dict | None:
        items = [item for item in self.evidence if item.proposition == proposition]
        if not items:
            return None
        by_polarity = {True: {}, False: {}}
        for item in items:
            current = by_polarity[item.polarity].get(item.source, 0.0)
            by_polarity[item.polarity][item.source] = max(current, item.source_score)
        scores = {polarity: sum(values.values()) for polarity, values in by_polarity.items()}
        source_counts = {polarity: len(values) for polarity, values in by_polarity.items()}
        if scores[True] and scores[False]:
            margin = abs(scores[True] - scores[False])
            status = "disputed" if margin < 0.75 else "provisional"
            accepted = None if status == "disputed" else scores[True] > scores[False]
        else:
            accepted = scores[True] > 0
            supporting = source_counts[accepted]
            status = "corroborated" if supporting >= 2 else "single_source"
        total = scores[True] + scores[False]
        confidence = 0.0 if total == 0 else max(scores.values()) / total
        return {
            "subject": proposition[0], "relation": proposition[1], "object": proposition[2],
            "scope": proposition[3], "status": status, "accepted_polarity": accepted,
            "confidence": round(confidence, 3),
            "supporting_sources": source_counts[True], "opposing_sources": source_counts[False],
            "support_score": round(scores[True], 3), "opposition_score": round(scores[False], 3),
            "citations": sorted({item.source_url for item in items}),
        }

    def report(self) -> dict:
        propositions = sorted({item.proposition for item in self.evidence})
        return {
            "beliefs": [self.belief(proposition) for proposition in propositions],
            "revisions": self.revisions,
            "evidence_count": len(self.evidence),
        }


class StoryConceptAgent:
    def __init__(self):
        self.extractor = ConceptExtractor()
        self.ledger = ConceptLedger()

    def ingest(self, source: str, url: str, score: float, sentences: list[str]) -> None:
        context_entities: set[str] = set()
        for sentence in sentences:
            lowered = sentence.lower()
            if "fox" in lowered:
                context_entities.add("fox")
            if "grape" in lowered:
                context_entities.add("grapes")
            for item in self.extractor.extract(sentence, source, url, score, context_entities):
                self.ledger.add(item)

    def learn_from_web(self, concept: str) -> dict:
        curriculum = StoryCurriculumAgent([WikisourceStories(), GutenbergStories()])
        search_report = curriculum.investigate(concept)
        for observation in curriculum.source_observations:
            self.ingest(observation["source"], observation["url"], observation["source_score"],
                        observation["sentences"])
        knowledge = self.ledger.report()
        return {
            "question": f"What actions and consequences are shared across stories about {concept}?",
            "generated_query": search_report["generated_query"],
            "sources": search_report["sources_found"],
            "concept_knowledge": knowledge,
            "cited_conclusions": [self.render_conclusion(belief) for belief in knowledge["beliefs"]],
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    @staticmethod
    def render_conclusion(belief: dict) -> dict:
        polarity = belief["accepted_polarity"]
        if polarity is None:
            claim = (f"Whether {belief['subject']} {belief['relation']} {belief['object']} "
                     f"is disputed in scope {belief['scope']}.")
            uncertainty = "opposing evidence is too close to choose"
        else:
            negative = "does not " if polarity is False else ""
            claim = f"{belief['subject']} {negative}{belief['relation']} {belief['object']} ({belief['scope']})."
            uncertainty = {
                "single_source": "only one independent source supports this",
                "corroborated": "corroborated but still falsifiable",
                "provisional": "one side currently outweighs live counterevidence",
            }.get(belief["status"], "unresolved")
        return {
            "claim": claim, "status": belief["status"], "confidence": belief["confidence"],
            "uncertainty": uncertainty, "citations": belief["citations"],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("concept", nargs="?", default="fox grapes")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = StoryConceptAgent().learn_from_web(args.concept)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
