#!/usr/bin/env python3
"""Ground an ambiguous Japanese surface in child-story observations without an LLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from japanese_boundaries_v18 import (
    JapaneseBoundaryAgent, JapaneseWikipedia, JapaneseWikisource, JapaneseWiktionary, _json,
)
from local_candidate_helper import OllamaCandidateHelper
from kanjipedia_reference_v22 import KanjipediaReference
from web_cache import WEB_CACHE


USER_AGENT = "AI_Noise/0.19 (read-only research; https://github.com/Noise101/AI_Noise)"
WIKI_API = "https://ja.wiktionary.org/w/api.php"
WIKIPEDIA_API = "https://ja.wikipedia.org/w/api.php"
STOP_FEATURES = {
    "もの", "こと", "よう", "など", "これ", "それ", "ある", "する", "いる", "なる",
    "れる", "られる", "ため", "また", "その", "この", "から", "である", "つる",
    "他", "物", "一部", "相手", "形", "状態",
}


def _clean_wikitext(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"'{2,}", "", text)
    return " ".join(text.split())


def _features(text: str) -> set[str]:
    chunks = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]+", text)
    features = set()
    for chunk in chunks:
        parts = re.split(r"(?:から|まで|より|など|として|について|の|が|を|に|で|と|や)", chunk)
        for part in parts:
            part = part.strip()
            if part and part not in STOP_FEATURES:
                features.add(part)
    return features


@dataclass
class JapaneseSense:
    label: str
    definition: str
    features: set[str]
    citations: list[str] = field(default_factory=list)
    reference_hashes: list[str] = field(default_factory=list)


class JapaneseWiktionarySenseSource:
    def lookup(self, form: str) -> list[JapaneseSense]:
        params = urllib.parse.urlencode({
            "action": "parse", "page": form, "prop": "wikitext", "format": "json", "formatversion": 2,
        })
        parsed = _json(f"{WIKI_API}?{params}").get("parse", {})
        wikitext = parsed.get("wikitext", "")
        headings = list(re.finditer(r"^===.*?:([^=\n]+)===$", wikitext, flags=re.MULTILINE))
        senses = []
        for index, match in enumerate(headings):
            label = _clean_wikitext(match.group(1)).strip()
            if not label or len(label) > 12:
                continue
            end = headings[index + 1].start() if index + 1 < len(headings) else len(wikitext)
            body = wikitext[match.end():end]
            definitions = []
            for line in body.splitlines():
                if line.startswith("#") and not line.startswith(("#:", "#*")):
                    cleaned = _clean_wikitext(line.lstrip("# "))
                    if cleaned:
                        definitions.append(cleaned)
            if not definitions:
                continue
            definition = " ".join(definitions[:4])
            senses.append(JapaneseSense(
                label, definition, _features(definition),
                ["https://ja.wiktionary.org/wiki/" + urllib.parse.quote(form)],
                [hashlib.sha256(definition.encode()).hexdigest()],
            ))
        return senses


class JapaneseWikipediaSenseSource:
    def enrich(self, sense: JapaneseSense) -> JapaneseSense:
        params = urllib.parse.urlencode({
            "action": "query", "titles": sense.label, "redirects": 1,
            "prop": "extracts|info|pageprops", "explaintext": 1, "exintro": 1,
            "inprop": "url", "format": "json", "formatversion": 2,
        })
        try:
            query = _json(f"{WIKIPEDIA_API}?{params}").get("query", {})
        except Exception:
            return sense
        pages = query.get("pages", [])
        if not pages or pages[0].get("missing") is True:
            return sense
        page = pages[0]
        extract = page.get("extract", "")
        if extract:
            sense.features.update(_features(extract))
            sense.citations.append(page.get("fullurl", "https://ja.wikipedia.org/wiki/" + urllib.parse.quote(page["title"])))
            sense.reference_hashes.append(hashlib.sha256(extract.encode()).hexdigest())
        return sense


class ContextSenseLedger:
    def __init__(self, senses: list[JapaneseSense]):
        self.senses = senses
        self.context_scores: Counter[str] = Counter()
        self.observations: list[dict] = []
        self.revisions: list[dict] = []

    def observe(self, context: str, source_url: str) -> dict:
        before = self.belief().get("accepted_sense")
        unique_counts = Counter(feature for sense in self.senses for feature in sense.features)
        additions = {}
        matched = {}
        for sense in self.senses:
            hits = sorted({feature for feature in sense.features
                           if (len(feature) >= 2 or any('\u3400' <= char <= '\u9fff' for char in feature))
                           and feature in context},
                          key=lambda feature: (-len(feature), feature))
            nonoverlap = []
            for feature in hits:
                if not any(feature in kept for kept in nonoverlap):
                    nonoverlap.append(feature)
            score = sum(min(2, context.count(feature)) *
                        (1.0 if unique_counts[feature] == 1 else 0.2)
                        for feature in nonoverlap[:12])
            if score:
                self.context_scores[sense.label] += score
            additions[sense.label] = round(score, 3)
            matched[sense.label] = nonoverlap[:12]
        observation = {"source": source_url, "sha256": hashlib.sha256(context.encode()).hexdigest(),
                       "score_additions": additions, "matched_features": matched}
        self.observations.append(observation)
        after = self.belief().get("accepted_sense")
        if before and after and before != after:
            self.revisions.append({"before": before, "after": after, "trigger_source": source_url,
                                   "reason": "new observable context changed the strongest sense"})
        return observation

    def belief(self) -> dict:
        ranked = sorted(self.senses, key=lambda sense: (-self.context_scores[sense.label], sense.label))
        if not ranked:
            return {"status": "unknown", "accepted_sense": None, "alternatives": []}
        top_score = self.context_scores[ranked[0].label]
        second_score = self.context_scores[ranked[1].label] if len(ranked) > 1 else 0
        margin = top_score - second_score
        accepted = ranked[0].label if top_score >= 2 and margin >= 1 else None
        status = "context_grounded_provisional" if accepted else "ambiguous"
        return {"status": status, "accepted_sense": accepted, "confidence_margin": round(margin, 3),
                "alternatives": [{"sense": sense.label, "context_score": round(self.context_scores[sense.label], 3),
                                  "definition": sense.definition, "citations": sense.citations,
                                  "reference_hashes": sense.reference_hashes}
                                 for sense in ranked]}

    def report(self) -> dict:
        return {"belief": self.belief(), "observations": self.observations, "revisions": self.revisions}


def run(seed_concept: str, use_local_helper: bool = False) -> dict:
    story_source = JapaneseWikisource()
    boundary_agent = JapaneseBoundaryAgent(
        story_source, [JapaneseWiktionary(), JapaneseWikipedia(), KanjipediaReference()])
    boundary = boundary_agent.learn(seed_concept, candidate_limit=15, target_words=2)
    ambiguous = next((item for item in boundary.get("accepted_words", [])
                      if item.get("meaning_status") == "ambiguous_reference"), None)
    if not ambiguous:
        return {"boundary_learning": boundary, "sense_grounding": None,
                "reason": "no corroborated boundary with ambiguous meaning"}
    story = story_source.search(boundary["generated_query"])
    senses = JapaneseWiktionarySenseSource().lookup(ambiguous["form"])
    wikipedia = JapaneseWikipediaSenseSource()
    senses = [wikipedia.enrich(sense) for sense in senses]
    ledger = ContextSenseLedger(senses)
    ledger.observe(story.text, story.url)
    belief = ledger.belief()
    accepted = next((sense for sense in senses if sense.label == belief.get("accepted_sense")), None)
    conclusion = {
        "claim": (f"この物語の『{ambiguous['form']}』は暫定的に『{belief['accepted_sense']}』を指す。"
                  if belief.get("accepted_sense") else f"この物語の『{ambiguous['form']}』の意味は未確定。"),
        "status": belief["status"], "confidence_margin": belief.get("confidence_margin", 0),
        "citations": [] if not accepted else accepted.citations,
        "falsification": "competing-sense features in new context can lower or replace this interpretation",
    }
    local_proposals = None
    if use_local_helper:
        proposal = OllamaCandidateHelper().propose_japanese_senses(ambiguous["form"], story.text)
        local_proposals = None if proposal is None else proposal.__dict__
    return {"boundary_learning": boundary,
            "sense_grounding": {"surface": ambiguous["form"], **ledger.report(),
                                "cited_conclusion": conclusion,
                                "local_unverified_proposals": local_proposals},
            "web_usage": WEB_CACHE.stats(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("concept", nargs="?", default="きつね つる")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--local-helper", action="store_true")
    args = parser.parse_args()
    report = run(args.concept, args.local_helper)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.summary:
        grounding = report.get("sense_grounding") or {}
        belief = grounding.get("belief", {})
        local = grounding.get("local_unverified_proposals") or {}
        print(json.dumps({"surface": grounding.get("surface"), "status": belief.get("status"),
                          "accepted": belief.get("accepted_sense"),
                          "confidence_margin": belief.get("confidence_margin"),
                          "citation_count": len(grounding.get("cited_conclusion", {}).get("citations", [])),
                          "local_unverified_candidates": len(local.get("candidates", [])),
                          "web_usage": report.get("web_usage")}, ensure_ascii=False))
    else:
        print(rendered)


if __name__ == "__main__":
    main()
