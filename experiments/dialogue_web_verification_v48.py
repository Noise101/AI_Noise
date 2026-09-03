"""Read-only independent web observations for hypotheses raised in local dialogue."""

from __future__ import annotations

import hashlib
import re
import time
import urllib.parse
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from web_cache import WEB_CACHE, NetworkBudgetExceeded


USER_AGENT = "AI_Noise/0.48 (read-only hypothesis verification; https://github.com/Noise101/AI_Noise)"
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@dataclass
class Observation:
    source: str
    url: str
    passage: str


class ObservationSource(Protocol):
    name: str
    def search(self, expression: str) -> list[Observation]: ...


class MediaWikiObservationSource:
    def __init__(self, host: str, name: str):
        self.host, self.name = host, name

    def search(self, expression: str) -> list[Observation]:
        params = urllib.parse.urlencode({
            "action": "query", "generator": "search", "gsrsearch": f'"{expression}"',
            "gsrnamespace": 0, "gsrlimit": 3, "prop": "extracts", "explaintext": 1,
            "exsentences": 5, "format": "json", "formatversion": 2,
        })
        api_url = f"https://{self.host}/w/api.php?{params}"
        try:
            pages = WEB_CACHE.get_json(api_url, USER_AGENT).get("query", {}).get("pages", [])
        except (Exception, NetworkBudgetExceeded):
            return []
        result = []
        for page in pages:
            passage = " ".join(str(page.get("extract", "")).split())
            if expression.lower() not in passage.lower():
                continue
            title = page.get("title", "")
            url = f"https://{self.host}/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            result.append(Observation(self.name, url, passage[:2000]))
        return result


def empty_verification_memory() -> dict:
    return {"version": 48, "expressions": {}, "observations": [], "summary": {},
            "policy": {"read_only": True, "local_llm_is_evidence": False,
                       "final_judgment": "Noise evidence rule"}}


def _contexts(expression: str, passage: str) -> list[dict]:
    words = WORD.findall(passage.lower())
    target = WORD.findall(expression.lower())
    if not target:
        return []
    found = []
    for index in range(len(words) - len(target) + 1):
        if words[index:index + len(target)] != target:
            continue
        found.append({"before": words[max(0, index - 5):index],
                      "expression": " ".join(target),
                      "after": words[index + len(target):index + len(target) + 5]})
    return found[:5]


def verify_dialogue_hypothesis(turn: dict, memory: dict,
                               sources: list[ObservationSource] | None = None,
                               network_budget: int = 2) -> dict:
    """Let deterministic Noise rules judge independent observations, never the partner."""
    expression = str(turn.get("unknown_expression") or "").strip().lower()
    focus = str(turn.get("hypothesis_focus") or "").strip().lower()
    if not expression or len(expression) > 60:
        return {"status": "not_searchable", "expression": expression,
                "final_judgment_made_by": "noise_evidence_rule_v1"}
    records = memory.setdefault("expressions", {})
    previous = records.get(expression, {})
    if previous.get("independent_sources", 0) >= 2:
        return {"status": "already_independently_observed", "expression": expression,
                "hypothesis_status": previous.get("hypothesis_status", "unresolved"),
                "final_judgment_made_by": "noise_evidence_rule_v1"}
    sources = sources or [
        MediaWikiObservationSource("en.wikisource.org", "English Wikisource"),
        MediaWikiObservationSource("simple.wikipedia.org", "Simple English Wikipedia"),
    ]
    WEB_CACHE.set_network_budget(max(0, network_budget))
    observations = []
    for source in sources:
        observations.extend(source.search(expression))
    independent_sources = sorted({item.source for item in observations})
    contexts = [context for item in observations for context in _contexts(expression, item.passage)]
    neighboring = [word for context in contexts for word in context["before"] + context["after"]]
    source_focus_support = {
        item.source for item in observations
        if any(focus in context["before"] + context["after"]
               for context in _contexts(expression, item.passage))
    } if focus else set()
    following = {tuple(context["after"][:2]) for context in contexts if context["after"]}
    if len(source_focus_support) >= 2:
        hypothesis_status = "supported_candidate_not_meaning_proof"
    elif len(independent_sources) >= 2 and focus and not source_focus_support:
        hypothesis_status = "rejected_as_overspecific"
    else:
        hypothesis_status = "unresolved"
    status = ("usage_corroborated" if len(independent_sources) >= 2 else
              "usage_observed_single_source" if independent_sources else "no_independent_observation")
    result = {
        "verification_id": hashlib.sha256(
            f"{expression}\n{focus}\n{len(memory.get('observations', []))}".encode()).hexdigest()[:20],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "expression": expression, "hypothesis_focus": focus,
        "status": status, "hypothesis_status": hypothesis_status,
        "independent_sources": len(independent_sources), "source_names": independent_sources,
        "source_urls": sorted({item.url for item in observations}),
        "contexts": contexts[:12], "context_diversity": len(following),
        "common_neighbors": Counter(neighboring).most_common(8),
        "partner_claim_used_as_evidence": False,
        "meaning_committed": False,
        "final_judgment_made_by": "noise_evidence_rule_v1",
    }
    memory.setdefault("observations", []).append(result)
    memory["observations"] = memory["observations"][-2000:]
    old_attempts = previous.get("attempts", 0)
    records[expression] = {**result, "attempts": old_attempts + 1}
    values = list(records.values())
    memory["summary"] = {
        "expressions_investigated": len(values),
        "independently_observed": sum(item.get("independent_sources", 0) >= 2 for item in values),
        "supported_candidates": sum(item.get("hypothesis_status") ==
                                    "supported_candidate_not_meaning_proof" for item in values),
        "rejected_overspecific": sum(item.get("hypothesis_status") ==
                                    "rejected_as_overspecific" for item in values),
        "unresolved": sum(item.get("hypothesis_status") == "unresolved" for item in values),
        "local_llm_claims_accepted_as_fact": 0,
    }
    return result
