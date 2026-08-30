#!/usr/bin/env python3
"""Read-only autonomous web learning over Wikidata and Wikipedia APIs."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


USER_AGENT = "AI_Noise/0.11 (read-only research; https://github.com/Noise101/AI_Noise)"


class WebProvider(Protocol):
    def search(self, query: str, limit: int = 5) -> list[dict]: ...
    def entity(self, entity_id: str) -> dict: ...
    def wikipedia_summary(self, title: str, language: str = "en") -> dict | None: ...


class WikimediaProvider:
    API = "https://www.wikidata.org/w/api.php"

    @staticmethod
    def _json(url: str) -> dict:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        params = urllib.parse.urlencode({
            "action": "wbsearchentities", "search": query, "language": "en",
            "uselang": "en", "type": "item", "limit": limit, "format": "json",
        })
        return self._json(f"{self.API}?{params}").get("search", [])

    def entity(self, entity_id: str) -> dict:
        params = urllib.parse.urlencode({
            "action": "wbgetentities", "ids": entity_id,
            "props": "labels|descriptions|claims|sitelinks", "languages": "en|ja", "format": "json",
        })
        return self._json(f"{self.API}?{params}").get("entities", {}).get(entity_id, {})

    def wikipedia_summary(self, title: str, language: str = "en") -> dict | None:
        encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
        url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        try:
            return self._json(url)
        except Exception:
            return None


@dataclass
class Evidence:
    source_url: str
    source_type: str
    strength: float
    references: int = 0


@dataclass
class Claim:
    subject: str
    property_id: str
    value: str
    rank: str
    evidence: Evidence
    accepted: bool = False

    @property
    def score(self) -> float:
        rank_bonus = {"preferred": 0.25, "normal": 0.10, "deprecated": -0.40}.get(self.rank, 0.0)
        return self.evidence.strength + rank_bonus + min(0.25, self.evidence.references * 0.05)


@dataclass
class EntityKnowledge:
    entity_id: str
    label: str
    description: str
    sources: list[str] = field(default_factory=list)
    explored: bool = False


class KnowledgeLedger:
    def __init__(self):
        self.entities: dict[str, EntityKnowledge] = {}
        self.claims: list[Claim] = []
        self.conflicts: list[dict] = []

    def add_claim(self, claim: Claim) -> None:
        peers = [c for c in self.claims if c.subject == claim.subject and c.property_id == claim.property_id]
        self.claims.append(claim)
        candidates = peers + [claim]
        winner = max(candidates, key=lambda item: (item.score, item.value))
        for candidate in candidates:
            candidate.accepted = candidate is winner
        alternatives = sorted({candidate.value for candidate in candidates})
        if len(alternatives) > 1:
            self.conflicts.append({
                "subject": claim.subject, "property": claim.property_id,
                "values": alternatives, "accepted": winner.value,
                "reason": "highest evidence score; alternatives retained",
            })

    def accepted_claims(self) -> list[Claim]:
        return [claim for claim in self.claims if claim.accepted]


class AutonomousWebLearner:
    """Chooses the next entity from its own unresolved knowledge frontier."""

    def __init__(self, provider: WebProvider, fanout: int = 6):
        self.provider = provider
        self.fanout = fanout
        self.ledger = KnowledgeLedger()
        self.frontier: dict[str, float] = {}
        self.goals: list[dict] = []

    def seed(self, topic: str) -> str:
        results = self.provider.search(topic, limit=5)
        if not results:
            raise RuntimeError(f"No Wikidata entity found for {topic!r}")
        root = results[0]["id"]
        self.frontier[root] = 1.0
        self.goals.append({"kind": "resolve_topic", "query": topic, "selected": root})
        return root

    def choose_goal(self) -> str | None:
        candidates = [(priority, entity_id) for entity_id, priority in self.frontier.items()
                      if not self.ledger.entities.get(entity_id, EntityKnowledge(entity_id, "", "")).explored]
        if not candidates:
            return None
        _, selected = max(candidates, key=lambda item: (item[0], item[1]))
        label = self.ledger.entities.get(selected, EntityKnowledge(selected, selected, "")).label or selected
        self.goals.append({"kind": "fill_entity_gap", "entity": selected, "question": f"What is {label}, and what is it related to?"})
        return selected

    @staticmethod
    def _value_id(statement: dict) -> str | None:
        value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, dict) and "id" in value:
            return value["id"]
        if isinstance(value, (str, int, float)):
            return str(value)
        if isinstance(value, dict) and "time" in value:
            return str(value["time"])
        return None

    def explore(self, entity_id: str) -> None:
        data = self.provider.entity(entity_id)
        label = data.get("labels", {}).get("en", {}).get("value", entity_id)
        description = data.get("descriptions", {}).get("en", {}).get("value", "")
        entity_url = f"https://www.wikidata.org/wiki/{entity_id}"
        knowledge = self.ledger.entities.setdefault(entity_id, EntityKnowledge(entity_id, label, description))
        knowledge.label, knowledge.description, knowledge.explored = label, description, True
        knowledge.sources.append(entity_url)

        linked: list[tuple[float, str]] = []
        for property_id, statements in data.get("claims", {}).items():
            for statement in statements:
                value = self._value_id(statement)
                if value is None:
                    continue
                references = len(statement.get("references", []))
                strength = 0.70 if references else 0.48
                claim = Claim(entity_id, property_id, value, statement.get("rank", "normal"),
                              Evidence(entity_url, "wikidata", strength, references))
                self.ledger.add_claim(claim)
                if value.startswith("Q") and value[1:].isdigit():
                    linked.append((claim.score, value))

        sitelink = data.get("sitelinks", {}).get("enwiki", {}).get("title")
        if sitelink:
            summary = self.provider.wikipedia_summary(sitelink, "en")
            if summary and summary.get("content_urls", {}).get("desktop", {}).get("page"):
                knowledge.sources.append(summary["content_urls"]["desktop"]["page"])

        for score, linked_id in sorted(linked, reverse=True)[: self.fanout]:
            if linked_id not in self.ledger.entities or not self.ledger.entities[linked_id].explored:
                self.frontier[linked_id] = max(self.frontier.get(linked_id, 0.0), score)

    def learn(self, topic: str, steps: int = 8) -> dict:
        root = self.seed(topic)
        for step in range(steps):
            goal = self.choose_goal()
            if goal is None:
                break
            self.explore(goal)
            print(f"progress {step + 1}/{steps} explored={goal} frontier={len(self.frontier)}", flush=True)
        return self.report(root)

    def report(self, root: str) -> dict:
        root_entity = self.ledger.entities.get(root)
        return {
            "root": None if root_entity is None else asdict(root_entity),
            "entities_explored": sum(entity.explored for entity in self.ledger.entities.values()),
            "frontier_size": len(self.frontier),
            "accepted_claims": [
                {"subject": c.subject, "property": c.property_id, "value": c.value,
                 "confidence": round(min(1.0, c.score), 3), "source": c.evidence.source_url}
                for c in self.ledger.accepted_claims()
            ],
            "conflicts": self.ledger.conflicts,
            "goals": self.goals,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = AutonomousWebLearner(WikimediaProvider()).learn(args.topic, args.steps)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

