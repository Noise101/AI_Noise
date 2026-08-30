#!/usr/bin/env python3
"""Close the unknown-word loop with read-only dictionary and usage evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol

from developmental_language_v15 import MultiLevelLearningAgent
from web_cache import WEB_CACHE


USER_AGENT = "AI_Noise/0.16 (read-only research; https://github.com/Noise101/AI_Noise)"


class _DefinitionParser(HTMLParser):
    def __init__(self, require_english_heading: bool):
        super().__init__()
        self.require_english_heading = require_english_heading
        self.in_english = not require_english_heading
        self.heading_level = None
        self.heading_text: list[str] = []
        self.ol_depth = 0
        self.li_depth = 0
        self.current: list[str] = []
        self.definitions: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"h2", "h3"}:
            self.heading_level = tag
            self.heading_text = []
        if tag == "ol" and self.in_english:
            self.ol_depth += 1
        if tag == "li" and self.in_english and self.ol_depth == 1:
            self.li_depth = 1
            self.current = []
        elif tag == "li" and self.li_depth:
            self.li_depth += 1

    def handle_endtag(self, tag):
        if tag in {"h2", "h3"} and self.heading_level == tag:
            heading = " ".join(self.heading_text).strip().lower()
            if tag == "h2":
                if heading == "english":
                    self.in_english = True
                elif self.require_english_heading and self.in_english:
                    self.in_english = False
            self.heading_level = None
        if tag == "li" and self.li_depth:
            if self.li_depth == 1:
                text = " ".join(" ".join(self.current).split())
                text = re.sub(r"\[\s*\d+\s*\]", "", text)
                if text and text not in self.definitions:
                    self.definitions.append(text)
                self.current = []
            self.li_depth -= 1
        if tag == "ol" and self.in_english and self.ol_depth:
            self.ol_depth -= 1

    def handle_data(self, data):
        if self.heading_level:
            self.heading_text.append(data)
        elif self.li_depth == 1:
            self.current.append(data)


@dataclass
class DefinitionDocument:
    source: str
    url: str
    definitions: list[str]
    source_score: float

    def evidence(self) -> dict:
        content = "\n".join(self.definitions)
        return {"source": self.source, "url": self.url, "definitions": self.definitions,
                "source_score": self.source_score,
                "sha256": hashlib.sha256(content.encode()).hexdigest()}


class DefinitionSource(Protocol):
    name: str
    def lookup(self, word: str) -> DefinitionDocument | None: ...


class WiktionaryDefinitions:
    def __init__(self, host: str, name: str, score: float, require_english_heading: bool):
        self.host, self.name, self.score = host, name, score
        self.require_english_heading = require_english_heading

    def lookup(self, word: str) -> DefinitionDocument | None:
        params = urllib.parse.urlencode({
            "action": "parse", "page": word, "prop": "text", "format": "json", "formatversion": 2,
        })
        url = f"https://{self.host}/w/api.php?{params}"
        try:
            parsed = WEB_CACHE.get_json(url, USER_AGENT).get("parse", {})
        except Exception:
            return None
        parser = _DefinitionParser(self.require_english_heading)
        parser.feed(parsed.get("text", ""))
        definitions = [definition for definition in parser.definitions if len(definition.split()) <= 60][:12]
        if not definitions:
            return None
        page = f"https://{self.host}/wiki/{urllib.parse.quote(word)}"
        return DefinitionDocument(self.name, page, definitions, self.score)


@dataclass(frozen=True)
class SenseEvidence:
    sense: str
    source: str
    source_url: str
    evidence_type: str
    score: float
    detail_hash: str


class LexicalMeaningLedger:
    def __init__(self):
        self.evidence: list[SenseEvidence] = []
        self.revisions: list[dict] = []
        self.previous_winner: str | None = None

    def add(self, evidence: SenseEvidence) -> None:
        before_belief = self.belief() if self.evidence else {}
        before = before_belief.get("accepted_sense") or before_belief.get("leading_sense")
        self.evidence.append(evidence)
        after_belief = self.belief()
        after = after_belief.get("accepted_sense") or after_belief.get("leading_sense")
        if before and after and before != after:
            self.revisions.append({"before": before, "after": after,
                                   "trigger_source": evidence.source,
                                   "reason": "new definition or usage evidence changed the leading sense"})

    def belief(self) -> dict:
        scores = defaultdict(float)
        sources = defaultdict(set)
        citations = defaultdict(set)
        for item in self.evidence:
            key = (item.sense, item.source)
            scores[key] = max(scores[key], item.score)
            sources[item.sense].add(item.source)
            citations[item.sense].add(item.source_url)
        totals = {sense: sum(score for (candidate, _), score in scores.items() if candidate == sense)
                  for sense in sources}
        ranked = sorted(totals, key=lambda sense: (-totals[sense], sense))
        if not ranked:
            return {"status": "unknown", "accepted_sense": None, "alternatives": []}
        top = ranked[0]
        margin = totals[top] - (totals[ranked[1]] if len(ranked) > 1 else 0)
        status = "ambiguous" if len(ranked) > 1 and margin < 0.5 else (
            "corroborated" if len(sources[top]) >= 2 else "single_source")
        return {
            "status": status, "accepted_sense": None if status == "ambiguous" else top,
            "leading_sense": top, "confidence_margin": round(margin, 3),
            "alternatives": [{"sense": sense, "score": round(totals[sense], 3),
                              "sources": len(sources[sense]), "citations": sorted(citations[sense])}
                             for sense in ranked],
        }


class LexicalResearchAgent:
    SENSE_MARKERS = {
        "repetition": ("one more time", "once more", "another time", "repeated", "repetition"),
        "return_to_prior_state": ("back to", "previous place", "previous position", "as before"),
        "additional_occurrence": ("in addition", "also", "further"),
    }

    def __init__(self, sources: list[DefinitionSource]):
        self.sources = sources
        self.ledger = LexicalMeaningLedger()

    @classmethod
    def senses_from_definition(cls, definition: str) -> set[str]:
        lowered = definition.lower()
        return {sense for sense, markers in cls.SENSE_MARKERS.items()
                if any(marker in lowered for marker in markers)}

    def investigate(self, gap: dict, usage_links: dict[tuple[str, str], int]) -> dict:
        word, query = gap["form"], gap["query"]
        documents = []
        for source in self.sources:
            document = source.lookup(word)
            if not document:
                continue
            documents.append(document.evidence())
            for definition in document.definitions:
                for sense in self.senses_from_definition(definition):
                    self.ledger.add(SenseEvidence(
                        sense, document.source, document.url, "definition", document.source_score,
                        hashlib.sha256(definition.encode()).hexdigest(),
                    ))
        repeated_pattern = usage_links.get((word, "and"), 0) and usage_links.get(("and", word), 0)
        if repeated_pattern:
            self.ledger.add(SenseEvidence(
                "repetition", "observed_child_story_usage", "local:evidence-ledger", "usage", 0.65,
                hashlib.sha256(f"{word} and {word}".encode()).hexdigest(),
            ))
        return {
            "detected_gap": gap, "executed_query": query, "word": word,
            "definition_sources": documents, "meaning_belief": self.ledger.belief(),
            "belief_revisions": self.ledger.revisions,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def run(seed_concept: str) -> dict:
    developmental = MultiLevelLearningAgent()
    first = developmental.learn_from_web(seed_concept)
    gap = developmental.lexicon.lexical_gap()
    if not gap or gap["kind"] != "unknown_word_meaning":
        return {"developmental_learning": first, "lexical_research": None,
                "reason": "no searchable unknown whitespace-delimited word"}
    sources = [
        WiktionaryDefinitions("en.wiktionary.org", "English Wiktionary", 0.82, True),
        WiktionaryDefinitions("simple.wiktionary.org", "Simple English Wiktionary", 0.78, False),
    ]
    research = LexicalResearchAgent(sources).investigate(gap, developmental.lexicon.word_links)
    developmental.lexicon.update_meaning_hypothesis(gap["form"], research["meaning_belief"])
    return {"developmental_learning": first, "lexical_research": research,
            "updated_lexicon": developmental.lexicon.report()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("concept", nargs="?", default="fox grapes")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.concept)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
