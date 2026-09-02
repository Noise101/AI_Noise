#!/usr/bin/env python3
"""Reparse retained source sentences into a conservative, reversible experience layer."""

from __future__ import annotations

from collections import Counter, defaultdict

from narrative_event_v29 import NarrativeEventExtractor


def rebuild_verified_experience(audit_memory: dict) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in audit_memory.get("records", {}).values():
        if item.get("curriculum_admitted") is True and item.get("sentence"):
            grouped[(item.get("seed", ""), item.get("source_url", ""))].append(item)

    extractor = NarrativeEventExtractor("developmental_grounded")
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
        results = extractor.extract_sequence(sentences)
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
        "policy": "developmental_grounded",
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
