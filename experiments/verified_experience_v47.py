#!/usr/bin/env python3
"""Reparse retained source sentences into a conservative, reversible experience layer."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from narrative_event_v29 import NarrativeEventExtractor


def rebuild_verified_experience(audit_memory: dict,
                                policy: str = "developmental_grounded_18") -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in audit_memory.get("records", {}).values():
        if item.get("curriculum_admitted") is True and item.get("sentence"):
            grouped[(item.get("seed", ""), item.get("source_url", ""))].append(item)

    extractor = NarrativeEventExtractor(policy)
    transitions: dict[str, dict[str, int]] = {}
    coherent_transitions: dict[str, dict[str, int]] = {}
    contextual: dict[str, dict[str, int]] = {}
    event_counts = Counter()
    rejected = Counter()
    sequences = []
    accepted_sentences = 0
    for (seed, source_url), records in sorted(grouped.items()):
        ordered = sorted(records, key=lambda item: item.get("source_position", 0))
        sentences = [item["sentence"] for item in ordered]
        # A coordinate-clause sentence ("The fox saw the grapes and jumped.") is not
        # one ambiguous compound event: split it into its simple clauses first so each
        # can pass the per-clause developmental checks on its own merits, instead of
        # quarantining the whole sentence as outside_simple_clause.
        results = extractor.extract_multi_sequence(sentences)
        events = []
        for result in results:
            if result.accepted and result.event and result.quality >= 0.85:
                events.append(result.event.key)
                event_counts[result.event.key] += 1
                accepted_sentences += 1
            else:
                # A rejection breaks temporal continuity; text on either side is not adjacent evidence.
                if events:
                    sequences.append({"seed": seed, "source_url": source_url,
                                      "events": events})
                    events = []
                reason = ("low_quality:" if result.accepted else "") + result.reason
                rejected[reason] += 1
        if events:
            sequences.append({"seed": seed, "source_url": source_url, "events": events})

    for sequence in sequences:
        events = sequence["events"]
        for index in range(1, len(events)):
            prior, outcome = events[index - 1], events[index]
            bucket = transitions.setdefault(prior, {})
            bucket[outcome] = bucket.get(outcome, 0) + 1
            # A subject switch without an explicit discourse model is not evidence
            # that the first event predicts the second.
            if prior.split("|", 1)[0] != outcome.split("|", 1)[0]:
                continue
            coherent_bucket = coherent_transitions.setdefault(prior, {})
            coherent_bucket[outcome] = coherent_bucket.get(outcome, 0) + 1
            if index >= 2:
                if events[index - 2].split("|", 1)[0] != prior.split("|", 1)[0]:
                    continue
                context = f"{events[index - 2]}>>{prior}"
                contextual_bucket = contextual.setdefault(context, {})
                contextual_bucket[outcome] = contextual_bucket.get(outcome, 0) + 1

    return {
        "version": 47,
        "policy": policy,
        "event_counts": dict(event_counts),
        "transitions": transitions,
        "coherent_transitions": coherent_transitions,
        "contextual_transitions": contextual,
        "sequences": sequences,
        "summary": {
            "sources": len(grouped), "accepted_sentences": accepted_sentences,
            "events": sum(event_counts.values()), "unique_events": len(event_counts),
            "transition_observations": sum(sum(items.values()) for items in transitions.values()),
            "coherent_transition_observations": sum(
                sum(items.values()) for items in coherent_transitions.values()),
            "contextual_observations": sum(sum(items.values()) for items in contextual.values()),
            "quarantined_sentences": sum(rejected.values()),
            "quarantine_reasons": dict(rejected.most_common()),
        },
        "warning": "original audit is retained; only conservative reparses feed prediction",
    }


def evaluate_experience_profile(experience: dict) -> dict:
    """Score a parsing policy on whole unseen sources, separate from its extraction count."""
    train, test = [], []
    for sequence in experience.get("sequences", []):
        target = (test if hashlib.sha256(
            f"profile-source:{sequence.get('source_url', '')}".encode()).digest()[0] % 5 == 0
                  else train)
        events = sequence.get("events", [])
        for prior, outcome in zip(events, events[1:]):
            if prior.split("|", 1)[0] == outcome.split("|", 1)[0]:
                target.append((prior, outcome))
    choices: dict[str, Counter[str]] = defaultdict(Counter)
    outcomes = Counter()
    for prior, outcome in train:
        prior_action = prior.split("|", 2)[1]
        outcome_action = outcome.split("|", 2)[1]
        choices[prior_action][outcome_action] += 1
        outcomes[outcome_action] += 1
    fallback = outcomes.most_common(1)[0][0] if outcomes else None
    correct = baseline = covered = 0
    for prior, outcome in test:
        prior_action = prior.split("|", 2)[1]
        observed = outcome.split("|", 2)[1]
        prediction = choices[prior_action].most_common(1)[0][0] if prior_action in choices else fallback
        correct += prediction == observed
        baseline += fallback == observed
        covered += prior_action in choices
    total = len(test)
    return {"policy": experience.get("policy"), "correct": correct,
            "baseline_correct": baseline, "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "baseline_accuracy": round(baseline / total, 4) if total else 0.0,
            "coverage": round(covered / total, 4) if total else 0.0,
            "accepted_sentences": experience.get("summary", {}).get("accepted_sentences", 0)}


def select_experience_profile(audit_memory: dict, previous: dict | None = None) -> tuple[dict, dict]:
    """Let evidence choose sentence complexity; retain history and reverse bad choices."""
    previous = previous or {}
    policies = tuple(f"developmental_grounded_{limit}" for limit in range(10, 25, 2))
    experiences = {policy: rebuild_verified_experience(audit_memory, policy) for policy in policies}
    evaluations = [evaluate_experience_profile(experiences[policy]) for policy in policies]
    default_policy = "developmental_grounded_18"
    eligible = [item for item in evaluations if item["total"] >= 20]
    selected = max(eligible, key=lambda item: (
        item["correct"] - item["baseline_correct"], item["correct"], item["coverage"],
        -abs(int(item["policy"].rsplit("_", 1)[-1]) - 18)), default=None)
    if selected is None or selected["correct"] <= selected["baseline_correct"]:
        selected = next(item for item in evaluations if item["policy"] == default_policy)
        status = "safe_default_until_predictive_evidence"
    else:
        status = "selected_on_unseen_sources"
    revisions = list(previous.get("revisions", []))
    before = previous.get("selected_policy")
    if before and before != selected["policy"]:
        revisions.append({"before": before, "after": selected["policy"],
                          "reason": "whole-source holdout changed policy ranking",
                          "evidence_total": selected["total"]})
    policy_memory = {"version": 49, "selected_policy": selected["policy"],
                     "selection_status": status, "selected_evaluation": selected,
                     "evaluations": evaluations, "revisions": revisions[-100:],
                     "mutable_parameter": "maximum words in a developmental sentence",
                     "safety_invariants": ["read_only_sources", "no_remote_llm_teacher",
                                           "explicit_action_required", "original_audit_retained"]}
    return experiences[selected["policy"]], policy_memory
