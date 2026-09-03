#!/usr/bin/env python3
"""Convert independently sourced experiences into tested, revisable rules."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict


def _parts(event: str) -> tuple[str, str, str]:
    return tuple((event.split("|", 2) + ["", ""])[:3])


def _kind(value: str) -> str:
    if not value:
        return "none"
    if value in {"he", "she", "they", "it", "him", "her", "them"}:
        return "pronoun"
    return "entity"


def _source_holdout(source: str) -> bool:
    return hashlib.sha256(f"rule-source:{source}".encode()).digest()[0] % 5 == 0


def learn_experience_rules(verified: dict, previous: dict | None = None) -> dict:
    """Run structure -> compare -> hypothesize -> source-holdout test -> revise."""
    previous = previous or {}
    frames, train, test = [], [], []
    for sequence in verified.get("sequences", []):
        source = sequence.get("source_url", "")
        target = test if _source_holdout(source) else train
        events = sequence.get("events", [])
        for index, event in enumerate(events):
            subject, action, obj = _parts(event)
            frame = {"experience_id": hashlib.sha256(
                        f"{source}:{index}:{event}".encode()).hexdigest()[:20],
                     "source_id": source, "seed": sequence.get("seed", ""),
                     "event": {"actor": subject, "action": action, "object": obj},
                     "features": {"actor_type": _kind(subject),
                                  "object_type": _kind(obj.split("_")[0] if obj else ""),
                                  "negated": False},
                     "certainty": "audited_observed_text"}
            frames.append(frame)
        for prior, outcome in zip(events, events[1:]):
            if _parts(prior)[0] == _parts(outcome)[0]:
                target.append((source, prior, outcome))

    # Compare experiences sharing an action but differing in entity/object.
    groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for row in train:
        groups[_parts(row[1])[1]].append(row)
    comparisons = []
    for action, rows in groups.items():
        sources = {row[0] for row in rows}
        outcomes = {_parts(row[2])[1] for row in rows}
        if len(sources) >= 2 and len(outcomes) >= 2:
            comparisons.append({"shared": {"prior_action": action},
                                "different_outcomes": sorted(outcomes)[:8],
                                "independent_sources": len(sources),
                                "observations": len(rows)})

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    subjects: dict[tuple[str, str], set[str]] = defaultdict(set)
    global_outcomes = Counter()
    for source, prior, outcome in train:
        _, action, obj = _parts(prior)
        observed = _parts(outcome)[1]
        context = f"after_action:{action}|object_type:{_kind(obj.split('_')[0] if obj else '')}"
        counts[context][observed] += 1
        subjects[(context, observed)].add(_parts(prior)[0])
        global_outcomes[observed] += 1
    fallback = global_outcomes.most_common(1)[0][0] if global_outcomes else None

    feedback: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    trials, correct, baseline, covered = [], 0, 0, 0
    for source, prior, outcome in test:
        _, action, obj = _parts(prior)
        context = f"after_action:{action}|object_type:{_kind(obj.split('_')[0] if obj else '')}"
        observed = _parts(outcome)[1]
        prediction = counts[context].most_common(1)[0][0] if context in counts else fallback
        success = prediction == observed
        correct += success
        baseline += fallback == observed
        covered += context in counts
        if context in counts:
            feedback[(context, prediction)]["success" if success else "failure"] += 1
        trials.append({"source_id": source, "context": context, "prediction": prediction,
                       "observed": observed, "correct": success})

    old_rules = {item.get("rule_id"): item for item in previous.get("rules", [])}
    rules, revisions = [], list(previous.get("revision_history", []))
    for context, outcomes in counts.items():
        prediction, support = outcomes.most_common(1)[0]
        result = feedback[(context, prediction)]
        tested, successes, failures = result["success"] + result["failure"], result["success"], result["failure"]
        reliability = (successes + 1) / (tested + 2)
        independent = len(subjects[(context, prediction)])
        status = ("reusable" if tested >= 5 and independent >= 3 and reliability >= .7 else
                  "weakened" if failures >= 3 and reliability < .5 else "tentative")
        rule_id = hashlib.sha256(f"{context}->{prediction}".encode()).hexdigest()[:16]
        rule = {"rule_id": rule_id, "conditions": context, "prediction": prediction,
                "training_support": support, "independent_entities": independent,
                "holdout_successes": successes, "holdout_failures": failures,
                "reliability": round(reliability, 4), "status": status,
                "known_counterexamples": [x for x in trials if x["context"] == context and not x["correct"]][:5],
                "alternatives": [{"prediction": key, "support": value}
                                 for key, value in outcomes.most_common(4)[1:]]}
        before = old_rules.get(rule_id, {}).get("status")
        if before and before != status:
            revisions.append({"rule_id": rule_id, "before": before, "after": status,
                              "reason": "whole-source holdout evidence changed"})
        rules.append(rule)
    rules.sort(key=lambda x: (x["status"] != "reusable", -x["training_support"], x["rule_id"]))
    total = len(test)
    lift = correct - baseline
    material = total >= 20 and lift >= max(5, int(total * .10 + .999)) and covered / max(1, total) >= .2
    failed = sorted((r for r in rules if r["holdout_failures"]),
                    key=lambda r: (-r["holdout_failures"], r["rule_id"]))
    next_target = None
    if failed:
        item = failed[0]
        action = item["conditions"].split("|", 1)[0].split(":", 1)[-1]
        alternatives = [x["prediction"] for x in item["alternatives"][:2]]
        terms = list(dict.fromkeys(x for x in [action, item["prediction"], *alternatives]
                                   if x and x not in {"said", "say", "was", "were", "is"}))
        next_target = {"seed": " ".join(terms),
                       "reason": "seek an independent boundary case for a weakened rule",
                       "rule_id": item["rule_id"]}
    summary = {"structured_experiences": len(frames), "comparison_groups": len(comparisons),
               "candidate_rules": len(rules),
               "reusable_rules": sum(r["status"] == "reusable" for r in rules),
               "weakened_rules": sum(r["status"] == "weakened" for r in rules),
               "evaluation": {"correct": correct, "baseline_correct": baseline, "total": total,
                              "coverage": round(covered / total, 4) if total else 0.0,
                              "material_lift": material},
               "next_learning_target": next_target}
    return {"version": 50, "frames": frames[-5000:], "comparisons": comparisons[:1000],
            "rules": rules[:5000], "trials": trials[-2000:],
            "revision_history": revisions[-1000:], "summary": summary,
            "invariants": ["whole_source_holdout", "local_llm_has_zero_evidence_credit",
                           "failed_rules_are_retained", "causal_credit_is_false"],
            "causal_credit": False}
