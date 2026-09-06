#!/usr/bin/env python3
"""Point curiosity at what the model gets wrong, not at what is frequent.

The frequency-driven curiosity ledger fixates on closed-class phrase bigrams
("in the", "to the") -- unsatisfiable by design, so their pressure climbs
without bound.  This module:

  * `deprioritise_syntactic_curiosity` -- zeroes the pressure on closed-class
    phrase gaps (syntax, not a groundable concept) and caps the ledger size.
  * `active_learning_targets` -- turns the frozen-benchmark model's held-out
    misses into search seeds: the argument frames it predicts wrong most often
    become "find more stories with this pattern" requests.

No pretrained model; operates on the plain event_structure_v1 report.
"""

from __future__ import annotations

import re
from collections import Counter

WORD = re.compile(r"[A-Za-z]+")

FUNCTION_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "and", "or", "but", "if",
    "so", "then", "than", "as", "of", "to", "in", "on", "at", "by", "for", "with",
    "from", "into", "onto", "over", "under", "up", "down", "out", "off", "about",
    "he", "she", "it", "they", "we", "you", "i", "him", "her", "them", "us",
    "his", "her", "its", "their", "our", "your", "my",
    "is", "are", "was", "were", "be", "been", "being", "am", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "can", "may", "might",
    "must", "not", "no", "nor", "there", "here", "when", "where", "who", "which",
    "what", "why", "how", "all", "any", "some", "one", "two", "very", "too",
}


_QUOTED = re.compile(r'["“‘\']([^"”’\']{1,40})["”’\']')


def _phrase_text(entry: dict, gap_id: str) -> str:
    """The bare phrase behind a curiosity gap, unwrapping the query template
    ('"of the" phrase meaning simple English') and the 'phrase:' gap-id prefix."""
    match = _QUOTED.search(entry.get("query", ""))
    if match:
        return match.group(1)
    if ":" in gap_id:
        return gap_id.split(":", 1)[1]
    return entry.get("query", "")


def is_closed_class_phrase(query: str) -> bool:
    words = [w.lower() for w in WORD.findall(query or "")]
    return bool(words) and all(word in FUNCTION_WORDS for word in words)


def deprioritise_syntactic_curiosity(ledger: dict[str, dict], max_entries: int = 4000) -> dict:
    """Mutate `ledger` in place: retire closed-class phrase gaps, cap size."""
    retired = 0
    for gap_id, entry in ledger.items():
        if entry.get("status") != "wanting_to_know" or entry.get("layer") not in {"phrase", "word"}:
            continue
        if is_closed_class_phrase(_phrase_text(entry, gap_id)):
            entry["pressure"] = 0.0
            entry["status"] = "syntactic_not_a_learnable_concept"
            retired += 1
    if len(ledger) > max_entries:
        ranked = sorted(ledger.items(),
                        key=lambda kv: (kv[1].get("pressure", 0.0),
                                        kv[1].get("last_seen_cycle", 0)), reverse=True)
        keep = dict(ranked[:max_entries])
        evicted = len(ledger) - len(keep)
        ledger.clear()
        ledger.update(keep)
    else:
        evicted = 0
    return {"retired_syntactic_gaps": retired, "evicted_low_pressure_gaps": evicted,
            "ledger_size": len(ledger)}


def _seed_from_tokens(tokens: list[str], visited: set[str]) -> str | None:
    seed = " ".join(dict.fromkeys(
        t for t in tokens if t and t not in FUNCTION_WORDS and len(t) >= 3))[:60].strip()
    if seed and len(seed.split()) >= 2 and seed not in visited:
        return seed
    return None


def active_learning_targets(event_structure: dict, visited: set[str],
                            limit: int = 4) -> list[dict]:
    """Rank the model's held-out misses by how often each argument/confusion
    pattern recurs, and emit a search seed for the worst few."""
    counterexamples = event_structure.get("counterexamples", [])
    if not counterexamples:
        return []
    confusion = Counter((trial.get("subject", ""), trial.get("object", ""),
                         trial.get("predicted", ""), trial.get("observed", ""))
                        for trial in counterexamples)
    targets: list[dict] = []
    for (subject, obj, predicted, observed), count in confusion.most_common():
        if count < 2:
            break
        seed = _seed_from_tokens([observed, subject, obj], visited)
        if not seed:
            continue
        targets.append({
            "seed": seed, "score": 6.0,
            "reason": "seek more events with an argument frame the model predicts wrong",
            "parent_url": None,
            "confusion": {"subject": subject, "object": obj,
                          "predicted": predicted, "observed": observed, "count": count}})
        if len(targets) >= limit:
            break
    return targets
