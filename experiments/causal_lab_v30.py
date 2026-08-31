#!/usr/bin/env python3
"""Active causal-mechanism check in a controllable procedural micro-world."""

from __future__ import annotations

import hashlib
import itertools


def run_lab(seed: str, budget: int = 6) -> dict:
    """Infer an unknown Boolean mechanism by choosing discriminating interventions."""
    worlds = list(itertools.product((0, 1), repeat=3))
    hypotheses = [(feature, invert) for feature in range(3) for invert in (False, True)]
    digest = hashlib.sha256(seed.encode()).digest()
    hidden = (digest[0] % 3, bool(digest[1] % 2))
    observations = []
    remaining = hypotheses[:]
    for _ in range(budget):
        if len(remaining) <= 1:
            break
        intervention = max(worlds, key=lambda values: min(
            sum((values[f] ^ inv) == value for f, inv in remaining)
            for value in (0, 1)))
        outcome = int(bool(intervention[hidden[0]]) ^ hidden[1])
        before = len(remaining)
        remaining = [(feature, invert) for feature, invert in remaining
                     if int(bool(intervention[feature]) ^ invert) == outcome]
        observations.append({"set_features": list(intervention), "observed_outcome": outcome,
                             "hypotheses_before": before, "hypotheses_after": len(remaining)})
        worlds.remove(intervention)
    learned = remaining[0] if len(remaining) == 1 else None
    return {"environment": "procedural controllable micro-world",
            "interventions": observations, "identified": learned is not None,
            "prediction_test": None if learned is None else learned == hidden,
            "world_knowledge_credit": 0,
            "warning": "tests active causal-learning machinery, not facts about stories or reality"}
