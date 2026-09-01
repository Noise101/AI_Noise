#!/usr/bin/env python3
"""Autonomous, read-only curriculum search for the v12 story learner."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol

from story_learning_v12 import Event, StoryLearner
from narrative_event_v29 import NarrativeEventExtractor
from web_cache import WEB_CACHE


USER_AGENT = "AI_Noise/0.13 (read-only research; https://github.com/Noise101/AI_Noise)"
SENTENCE = re.compile(r"(?<=[.!?])\s+")
ATOM = {"a": "http://www.w3.org/2005/Atom"}
BOILERPLATE = {
    "project gutenberg", "public domain", "copyright", "all rights reserved",
    "terms of use", "license", "located in the united states", "transcription",
    "proofread", "navigation menu", "download", "ebook",
    "sister projects", "wikipedia article", "illustrated by", "translated by",
    "versions of", "routledge and sons", "perry index",
}
NON_ACTIONS = {"and", "or", "of", "by", "for", "in", "on", "at", "from", "false", "domain"}
VERBS = {
    "am", "are", "is", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "say", "said", "see", "saw", "seen", "go", "went", "come",
    "came", "get", "got", "make", "made", "take", "took", "give", "gave", "find",
    "found", "know", "knew", "think", "thought", "turn", "turned", "eat", "ate",
    "drink", "drank", "jump", "jumped", "run", "ran", "fall", "fell", "grow", "grew",
    "want", "wanted", "try", "tried", "reach", "reached", "miss", "missed", "wait",
    "waited", "ask", "asked", "answer", "answered", "leave", "left", "learn", "learnt",
    "refuse", "refused", "place", "placed", "hang", "hung", "resort", "resorted",
    "sees", "jumps", "waits", "pushes", "eats", "falls", "grows", "shines", "sings",
}
METADATA_PREFIXES = {
    "title ", "author ", "translator ", "language ", "credits ", "most recently ",
    "three hundred ", "aesop's fables", "aesop s fables", "literally translated ",
}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"p", "div", "h1", "h2", "h3", "h4", "li", "br"}:
            self.parts.append(". ")
        if tag in {"script", "style", "table"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"p", "div", "h1", "h2", "h3", "h4", "li"}:
            self.parts.append(". ")
        if tag in {"script", "style", "table"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


@dataclass
class StoryDocument:
    source: str
    title: str
    url: str
    text: str
    query: str
    source_score: float
    rights_note: str

    def evidence(self) -> dict:
        return {
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "query": self.query,
            "source_score": self.source_score,
            "rights_note": self.rights_note,
            "sha256": hashlib.sha256(self.text.encode()).hexdigest(),
        }


class StorySource(Protocol):
    name: str
    def search(self, query: str) -> StoryDocument | None: ...


def _request(url: str) -> bytes:
    return WEB_CACHE.get_bytes(url, USER_AGENT)


class WikisourceStories:
    name = "Wikisource"
    API = "https://en.wikisource.org/w/api.php"

    def search(self, query: str) -> StoryDocument | None:
        params = urllib.parse.urlencode({
            "action": "query", "list": "search", "srsearch": query,
            "srnamespace": 0, "srlimit": 5, "format": "json", "formatversion": 2,
        })
        results = json.loads(_request(f"{self.API}?{params}")).get("query", {}).get("search", [])
        if not results:
            return None
        candidates = []
        for result in results:
            title = result["title"]
            parse_params = urllib.parse.urlencode({
                "action": "parse", "page": title, "prop": "text", "format": "json", "formatversion": 2,
            })
            parsed = json.loads(_request(f"{self.API}?{parse_params}")).get("parse", {})
            extractor = _TextExtractor()
            extractor.feed(parsed.get("text", ""))
            text = extractor.text()
            if len(text) >= 400:
                candidates.append((0 if "/" in title else 1, len(text), title, text))
        if not candidates:
            return None
        _, _, title, text = min(candidates)
        url = "https://en.wikisource.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="()/_'%")
        return StoryDocument(self.name, title, url, text, query, 0.82,
                             "repository describes works according to page-level license/public-domain notices")


class GutenbergStories:
    name = "Project Gutenberg"
    SEARCH = "https://www.gutenberg.org/ebooks/search.opds/"

    def search(self, query: str) -> StoryDocument | None:
        feed = ET.fromstring(_request(f"{self.SEARCH}?{urllib.parse.urlencode({'query': query})}"))
        entry = feed.find("a:entry", ATOM)
        if entry is None:
            return None
        title = entry.findtext("a:title", default="untitled", namespaces=ATOM)
        detail_link = entry.find("a:link[@type='application/atom+xml;profile=opds-catalog']", ATOM)
        if detail_link is None:
            return None
        detail_url = urllib.parse.urljoin(self.SEARCH, detail_link.attrib["href"])
        detail = ET.fromstring(_request(detail_url))
        epub_url = None
        page_url = None
        for link in detail.findall(".//a:link", ATOM):
            if link.attrib.get("type") == "application/epub+zip" and "noimages" in link.attrib.get("href", ""):
                epub_url = link.attrib["href"]
                break
        alternate = detail.find(".//a:link[@rel='alternate']", ATOM)
        if alternate is not None:
            page_url = urllib.parse.urljoin(detail_url, alternate.attrib["href"])
        if not epub_url:
            return None
        text_parts = []
        with zipfile.ZipFile(io.BytesIO(_request(epub_url))) as archive:
            for name in sorted(archive.namelist()):
                if name.lower().endswith((".xhtml", ".html", ".htm")):
                    extractor = _TextExtractor()
                    extractor.feed(archive.read(name).decode("utf-8", "ignore"))
                    text_parts.append(extractor.text())
        return StoryDocument(self.name, title, page_url or epub_url, " ".join(text_parts), query, 0.86,
                             "Project Gutenberg public-domain ebook repository; verify its ebook license page")


class StoryCurriculumAgent:
    def __init__(self, sources: list[StorySource], learner: StoryLearner | None = None):
        self.sources = sources
        self.learner = learner or StoryLearner()
        self.documents: list[StoryDocument] = []
        self.search_history: list[dict] = []
        self.visited_urls: set[str] = set()
        self.source_observations: list[dict] = []

    def detect_gap(self, seed_concept: str) -> dict:
        open_questions = [q for q in self.learner.why_questions if q.status == "open"]
        if open_questions:
            question = open_questions[-1]
            terms = [part for part in (question.context + "|" + question.outcome).split("|") if part]
            return {"kind": "unexplained_surprise", "terms": terms, "why": question.question}
        sparse = [event for event, count in self.learner.event_counts.items() if count < 2]
        if sparse:
            return {"kind": "weak_rule", "terms": sparse[0].split("|"), "why": "only one observation"}
        return {"kind": "new_child_concept", "terms": seed_concept.split(),
                "why": "no child-level experience for the requested concept"}

    @staticmethod
    def make_query(gap: dict) -> str:
        useful = [term.replace("_", " ") for term in gap["terms"] if len(term) > 2]
        return " ".join(useful[:5] + ["fable"])

    @staticmethod
    def select_passage(text: str, query: str, limit: int = 12) -> list[str]:
        sentences = []
        for raw in SENTENCE.split(text):
            sentence = raw.strip()
            lowered = sentence.lower()
            if not 3 <= len(sentence.split()) <= 35:
                continue
            if any(marker in lowered for marker in BOILERPLATE):
                continue
            if sum(character.isalpha() for character in sentence) < len(sentence) * 0.55:
                continue
            sentences.append(sentence)
        terms = {term.lower() for term in re.findall(r"[A-Za-z]+", query) if len(term) > 3 and term != "fable"}
        scored = []
        for index, sentence in enumerate(sentences):
            words = {word.lower() for word in re.findall(r"[A-Za-z]+", sentence)}
            scored.append((len(words & terms), min(35, len(sentence.split())), -index, index))
        if not scored or max(scored)[0] == 0:
            return []
        center = max(scored)[3]
        def heading(sentence: str) -> bool:
            tokens = re.findall(r"[A-Za-z]+", sentence)
            if not 2 <= len(tokens) <= 8 or any(mark in sentence for mark in '“”"'):
                return False
            titled = sum(token[0].isupper() or token.isupper() for token in tokens)
            return titled / len(tokens) >= 0.65

        start = center if heading(sentences[center]) else max(0, center - 2)
        end = min(len(sentences), start + limit)
        for index in range(center + 1, end):
            if heading(sentences[index]):
                end = index
                break
        passage = sentences[start:end]
        return [sentence for sentence in passage if not any(marker in sentence.lower() for marker in BOILERPLATE)]

    @staticmethod
    def parse_child_event(sentence: str) -> Event | None:
        """Find an explicit action in prose without a statistical language model."""
        return NarrativeEventExtractor().extract(sentence).event

    def investigate(self, seed_concept: str) -> dict:
        gap = self.detect_gap(seed_concept)
        query = self.make_query(gap)
        found = []
        for source in self.sources:
            document = source.search(query)
            status = "not_found"
            if document:
                passage = self.select_passage(document.text, query)
                if passage and document.url not in self.visited_urls:
                    extractions = NarrativeEventExtractor().extract_multi_sequence(passage)
                    parsed_events = [item.event for item in extractions
                                     if item.event and item.event.action not in NON_ACTIONS]
                    accepted_passage = [item.sentence for item in extractions
                                        if item.event and item.event.action not in NON_ACTIONS]
                    learned_events = [event.key for event in parsed_events if event]
                    if not learned_events:
                        self.search_history.append({"source": source.name, "query": query,
                                                    "status": "quality_rejected"})
                        continue
                    self.learner.observe_events(parsed_events)
                    self.documents.append(document)
                    self.visited_urls.add(document.url)
                    self.source_observations.append({
                        "source": document.source, "title": document.title, "url": document.url,
                        "source_score": document.source_score, "sentences": passage,
                    })
                    found.append({
                        **document.evidence(),
                        "sentences_used": len(accepted_passage),
                        "learned_events": learned_events,
                        "event_extraction_audit": [item.record() for item in extractions],
                        "passage_sha256": hashlib.sha256("\n".join(accepted_passage).encode()).hexdigest(),
                    })
                    status = "learned"
                elif document.url in self.visited_urls:
                    status = "duplicate_skipped"
            self.search_history.append({"source": source.name, "query": query, "status": status})
        return {
            "gap": gap,
            "generated_query": query,
            "sources_found": found,
            "independent_sources": len({item["source"] for item in found}),
            "learning": self.learner.report(),
            "conclusion": self._conclusion(found),
            "search_history": self.search_history,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def run_curriculum(self, seed_concept: str, rounds: int = 3) -> dict:
        cycles = []
        for _ in range(rounds):
            cycles.append(self.investigate(seed_concept))
        all_sources = [source.evidence() for source in self.documents]
        return {
            "seed_concept": seed_concept,
            "cycles": [{
                "gap": cycle["gap"], "generated_query": cycle["generated_query"],
                "sources_found": cycle["sources_found"], "conclusion": cycle["conclusion"],
            } for cycle in cycles],
            "unique_sources_read": len(self.visited_urls),
            "evidence_ledger": all_sources,
            "final_learning": self.learner.report(),
            "search_history": self.search_history,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    @staticmethod
    def _conclusion(found: list[dict]) -> dict:
        if len(found) < 2:
            return {"status": "uncertain", "reason": "fewer than two independent repositories", "citations": found}
        return {"status": "provisional", "reason": "two repositories were read; learned rules remain falsifiable",
                "citations": [{"title": item["title"], "url": item["url"], "sha256": item["sha256"]} for item in found]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("concept", nargs="?", default="actions and consequences")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = StoryCurriculumAgent([WikisourceStories(), GutenbergStories()]).run_curriculum(args.concept, args.rounds)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
