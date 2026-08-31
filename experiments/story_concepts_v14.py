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
from narrative_event_v29 import NarrativeEventExtractor


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
    """Derive reusable propositions from events, without story-specific entities."""

    def extract(self, sentence: str, source: str, url: str, score: float,
                context_entities: set[str] | None = None) -> list[ConceptEvidence]:
        digest = hashlib.sha256(sentence.encode()).hexdigest()
        event = NarrativeEventExtractor().extract(sentence).event
        if not event:
            return []
        obj = event.object.split("_")[-1] if event.object else "self"
        return [ConceptEvidence(event.subject, event.action, obj, True, "observed_event",
                                source, url, score, digest)]


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
            "learned_relation_groups": self.learn_relation_groups(),
            "revisions": self.revisions,
            "evidence_count": len(self.evidence),
        }

    def learn_relation_groups(self) -> list[dict]:
        """Propose relation equivalence from shared use, never from a built-in synonym list."""
        contexts: dict[str, set[tuple[str, str]]] = {}
        for item in self.evidence:
            contexts.setdefault(item.relation, set()).add((item.subject, item.object))
        groups = []
        relations = sorted(contexts)
        for index, left in enumerate(relations):
            for right in relations[index + 1:]:
                shared = contexts[left] & contexts[right]
                union = contexts[left] | contexts[right]
                similarity = len(shared) / len(union) if union else 0.0
                if len(shared) >= 2 and similarity >= 0.5:
                    groups.append({"relations": [left, right], "shared_contexts": len(shared),
                                   "similarity": round(similarity, 3),
                                   "status": "distributional_candidate",
                                   "warning": "similar use is not identical meaning; countercontexts may split this group"})
        return sorted(groups, key=lambda item: (-item["similarity"], item["relations"]))


class StoryConceptAgent:
    def __init__(self):
        self.extractor = ConceptExtractor()
        self.ledger = ConceptLedger()

    def ingest(self, source: str, url: str, score: float, sentences: list[str]) -> None:
        extractions = NarrativeEventExtractor().extract_sequence(sentences)
        for sentence, extraction in zip(sentences, extractions):
            if not extraction.event:
                continue
            event = extraction.event
            digest = hashlib.sha256(sentence.encode()).hexdigest()
            obj = event.object.split("_")[-1] if event.object else "self"
            self.ledger.add(ConceptEvidence(
                event.subject, event.action, obj, True,
                "observed_event", source, url, score, digest))

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
