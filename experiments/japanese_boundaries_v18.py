#!/usr/bin/env python3
"""Discover Japanese word boundaries from child stories and validate them on the web."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from story_web_curriculum_v13 import _TextExtractor


USER_AGENT = "AI_Noise/0.18 (read-only research; https://github.com/Noise101/AI_Noise)"
JAPANESE_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]+")


def _json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


@dataclass
class JapaneseStory:
    title: str
    url: str
    text: str
    query: str


class JapaneseStorySource(Protocol):
    def search(self, query: str) -> JapaneseStory | None: ...


class JapaneseWikisource:
    API = "https://ja.wikisource.org/w/api.php"

    def search(self, query: str) -> JapaneseStory | None:
        params = urllib.parse.urlencode({
            "action": "query", "list": "search", "srsearch": query,
            "srnamespace": 0, "srlimit": 5, "format": "json", "formatversion": 2,
        })
        results = _json(f"{self.API}?{params}").get("query", {}).get("search", [])
        candidates = []
        query_terms = [term for term in query.split() if term]
        for result in results:
            title = result["title"]
            parse_params = urllib.parse.urlencode({
                "action": "parse", "page": title, "prop": "text", "format": "json", "formatversion": 2,
            })
            parsed = _json(f"{self.API}?{parse_params}").get("parse", {})
            extractor = _TextExtractor()
            extractor.feed(parsed.get("text", ""))
            text = extractor.text()
            if len(text) >= 300:
                narrative_score = text.count("。") - 3 * text.count("パブリックドメイン")
                title_overlap = sum(term in title for term in query_terms)
                candidates.append((title_overlap, int("/" in title), narrative_score,
                                   -len(text), title, text))
        if not candidates:
            return None
        _, _, _, _, title, text = max(candidates)
        url = "https://ja.wikisource.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="/")
        return JapaneseStory(title, url, text, query)


@dataclass
class BoundaryCandidate:
    form: str
    count: int
    left_variety: int
    right_variety: int
    score: float


class BoundaryInducer:
    def __init__(self, max_length: int = 6):
        self.max_length = max_length
        self.counts: Counter[str] = Counter()
        self.left: dict[str, set[str]] = defaultdict(set)
        self.right: dict[str, set[str]] = defaultdict(set)

    def observe(self, text: str) -> None:
        for run in JAPANESE_RUN.findall(text):
            for size in range(2, min(self.max_length, len(run)) + 1):
                for index in range(len(run) - size + 1):
                    chunk = run[index:index + size]
                    self.counts[chunk] += 1
                    self.left[chunk].add(run[index - 1] if index else "<BOUNDARY>")
                    self.right[chunk].add(run[index + size] if index + size < len(run) else "<BOUNDARY>")

    def candidates(self, minimum_count: int = 2) -> list[BoundaryCandidate]:
        candidates = []
        for form, count in self.counts.items():
            if count < minimum_count:
                continue
            variety = len(self.left[form]) + len(self.right[form])
            score = count * (len(form) - 1) * (1 + math.log2(max(1, variety)))
            candidates.append(BoundaryCandidate(form, count, len(self.left[form]),
                                                len(self.right[form]), round(score, 3)))
        return sorted(candidates, key=lambda item: (-item.score, -len(item.form), item.form))


class JapaneseReference(Protocol):
    name: str
    def lookup(self, form: str) -> dict | None: ...


class JapaneseWiktionary:
    name = "Japanese Wiktionary"
    API = "https://ja.wiktionary.org/w/api.php"

    def lookup(self, form: str) -> dict | None:
        params = urllib.parse.urlencode({
            "action": "parse", "page": form, "prop": "text", "format": "json", "formatversion": 2,
        })
        try:
            parsed = _json(f"{self.API}?{params}").get("parse", {})
        except Exception:
            return None
        extractor = _TextExtractor()
        extractor.feed(parsed.get("text", ""))
        text = extractor.text()
        if len(text) < 20:
            return None
        return {"source": self.name, "url": "https://ja.wiktionary.org/wiki/" + urllib.parse.quote(form),
                "sha256": hashlib.sha256(text.encode()).hexdigest(), "text_length": len(text)}


class JapaneseWikipedia:
    name = "Japanese Wikipedia"
    API = "https://ja.wikipedia.org/w/api.php"

    def lookup(self, form: str) -> dict | None:
        params = urllib.parse.urlencode({
            "action": "query", "titles": form, "redirects": 1,
            "prop": "info|pageprops", "inprop": "url", "format": "json", "formatversion": 2,
        })
        try:
            query = _json(f"{self.API}?{params}").get("query", {})
        except Exception:
            return None
        pages = query.get("pages", [])
        if not pages or pages[0].get("missing") is True:
            return None
        page = pages[0]
        redirects = query.get("redirects", [])
        return {"source": self.name,
                "url": page.get("fullurl", "https://ja.wikipedia.org/wiki/" + urllib.parse.quote(page["title"])),
                "requested_title": form, "matched_title": page["title"], "redirects": redirects,
                "ambiguous": "disambiguation" in page.get("pageprops", {})}


class JapaneseBoundaryAgent:
    def __init__(self, story_source: JapaneseStorySource, references: list[JapaneseReference]):
        self.story_source = story_source
        self.references = references

    @staticmethod
    def make_query(seed_concept: str) -> str:
        return " ".join(part for part in seed_concept.split() if part)

    def learn(self, seed_concept: str, candidate_limit: int = 15) -> dict:
        query = self.make_query(seed_concept)
        story = self.story_source.search(query)
        if not story:
            return {"status": "no_story", "generated_query": query, "accepted_words": []}
        inducer = BoundaryInducer()
        inducer.observe(story.text)
        checked, accepted = [], []
        for candidate in inducer.candidates()[:candidate_limit]:
            evidence = []
            for reference in self.references:
                result = reference.lookup(candidate.form)
                if result:
                    evidence.append(result)
            status = "corroborated_boundary" if len(evidence) >= 2 else (
                "single_reference" if evidence else "unvalidated_chunk")
            meaning_status = "ambiguous_reference" if any(item.get("ambiguous") for item in evidence) else (
                "reference_entry_found_but_not_grounded" if evidence else "unknown")
            record = {**candidate.__dict__, "status": status, "reference_evidence": evidence,
                      "meaning_status": meaning_status,
                      "query": f'"{candidate.form}" 意味 やさしい例文'}
            checked.append(record)
            if len(evidence) >= 2:
                accepted.append(record)
        return {
            "status": "learned", "generated_query": query,
            "story_evidence": {"title": story.title, "url": story.url,
                               "sha256": hashlib.sha256(story.text.encode()).hexdigest()},
            "checked_candidates": checked, "accepted_words": accepted,
            "uncertainty": "a validated surface can still be inflected grammar or have multiple meanings",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("concept", nargs="?", default="きつね つる")
    parser.add_argument("--candidate-limit", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    agent = JapaneseBoundaryAgent(JapaneseWikisource(), [JapaneseWiktionary(), JapaneseWikipedia()])
    report = agent.learn(args.concept, args.candidate_limit)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
