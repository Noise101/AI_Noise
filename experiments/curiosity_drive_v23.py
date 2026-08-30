"""Persistent intrinsic pressure toward repeatedly encountered unknowns."""

from __future__ import annotations

import math


LAYER_DRIVE = {
    "character": 0.8,
    "word": 1.0,
    "phrase": 1.15,
    "conversation": 1.25,
    "why": 1.3,
    "concept": 1.2,
}


def observe_unknown(ledger: dict[str, dict], gap, cycle: int) -> dict:
    entry = ledger.setdefault(gap.gap_id, {
        "layer": gap.layer,
        "query": gap.query,
        "first_seen_cycle": cycle,
        "last_seen_cycle": cycle,
        "encounters": 0,
        "contexts_seen": 0,
        "status": "wanting_to_know",
        "resolution": None,
    })
    observed = max(1, int(gap.observations))
    if observed > entry["encounters"]:
        entry["contexts_seen"] += 1
        entry["last_seen_cycle"] = cycle
    entry["encounters"] = max(entry["encounters"], observed)
    entry["status"] = "wanting_to_know"
    entry["pressure"] = curiosity_pressure(entry, gap.uncertainty, cycle)
    return entry


def curiosity_pressure(entry: dict, uncertainty: float, cycle: int) -> float:
    encounters = max(1, entry.get("encounters", 1))
    age = max(0, cycle - entry.get("first_seen_cycle", cycle))
    contexts = max(1, entry.get("contexts_seen", 1))
    recurrence = 1 + math.log2(encounters + 1)
    unresolved_growth = 1 + min(age, 50) * 0.08
    context_growth = 1 + math.log2(contexts + 1) * 0.2
    drive = LAYER_DRIVE.get(entry.get("layer"), 1.0)
    return round(uncertainty * recurrence * unresolved_growth * context_growth * drive, 4)


def resolve_unknown(ledger: dict[str, dict], gap_id: str, resolution: str, cycle: int) -> None:
    entry = ledger.get(gap_id)
    if not entry:
        return
    entry.update({"status": "satisfied_for_now", "resolution": resolution,
                  "resolved_cycle": cycle, "pressure": 0.0})


def result_is_grounded(layer: str, result: dict) -> tuple[bool, str | None]:
    if layer in {"word", "phrase"}:
        belief = result.get("meaning_belief", {})
        accepted = belief.get("accepted_sense")
        return bool(accepted), accepted
    if layer == "why":
        answer = result.get("why", {})
        candidate = answer.get("candidate_cause")
        return bool(candidate), candidate
    if layer == "conversation":
        pattern = result.get("grounded_pattern")
        return bool(pattern), pattern
    return False, None
