#!/usr/bin/env python3
"""Select a transparent event-parser policy by unseen sequence prediction."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from narrative_event_v29 import NarrativeEventExtractor


POLICIES = ("baseline", "clause_head", "nearest_compact", "compact_roles")


def held_out(left_id: str, right_id: str) -> bool:
    return hashlib.sha256(f"parser:{left_id}->{right_id}".encode()).digest()[0] % 5 == 0


def predictive_key(event) -> str:
    obj = event.object.split("_")[-1] if event.object else "none"
    return f"{event.subject}|{event.action}|{obj}"


def outcome_key(event) -> str:
    obj = event.object.split("_")[-1] if event.object else "none"
    return f"{event.action}|{obj}"


def evaluate_policy(frames: dict, policy: str) -> dict:
    extractor = NarrativeEventExtractor(policy)
    parsed = {}
    roots = [identity for identity, frame in frames.items()
             if not frame.get("sequence", {}).get("previous_frame")]
    visited = set()
    for root in roots:
        identity, recent_subject, recent_object = root, None, None
        while identity and identity in frames and identity not in visited:
            visited.add(identity)
            frame = frames[identity]
            sentence = frame.get("observation", {}).get("sentence", "")
            result = extractor.extract(sentence, recent_subject, recent_object)
            if result.accepted and result.event:
                parsed[identity] = result.event
                recent_subject = result.event.subject
                object_tokens = result.event.object.split("_")
                recent_object = next((token for token in reversed(object_tokens)
                                      if token not in {"him", "her", "it", "them"}), recent_object)
            identity = frame.get("sequence", {}).get("next_frame")
    for identity, frame in frames.items():
        if identity in visited:
            continue
        result = extractor.extract(frame.get("observation", {}).get("sentence", ""))
        if result.accepted and result.event:
            parsed[identity] = result.event
    train, test = [], []
    for identity, frame in frames.items():
        next_id = frame.get("sequence", {}).get("next_frame")
        if identity not in parsed or next_id not in parsed:
            continue
        pair = (identity, next_id, parsed[identity], parsed[next_id])
        (test if held_out(identity, next_id) else train).append(pair)
    choices: dict[str, Counter[str]] = defaultdict(Counter)
    fallback = Counter()
    for _, _, prior, outcome in train:
        result = outcome_key(outcome)
        choices[predictive_key(prior)][result] += 1
        fallback[result] += 1
    common = fallback.most_common(1)[0][0] if fallback else None
    correct = baseline_correct = covered = 0
    trials = []
    for left_id, right_id, prior, outcome in test:
        context, observed = predictive_key(prior), outcome_key(outcome)
        predicted = choices[context].most_common(1)[0][0] if context in choices else common
        success = predicted == observed
        correct += success
        baseline_correct += common == observed
        covered += context in choices
        trials.append({"from_frame": left_id, "to_frame": right_id,
                       "context": context, "predicted": predicted,
                       "observed": observed, "correct": success})
    total = len(test)
    return {"policy": policy, "parsed_frames": len(parsed), "available_frames": len(frames),
            "parse_coverage": round(len(parsed) / len(frames), 4) if frames else 0.0,
            "correct": correct, "baseline_correct": baseline_correct, "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "baseline_accuracy": round(baseline_correct / total, 4) if total else 0.0,
            "prediction_coverage": round(covered / total, 4) if total else 0.0,
            "trials": trials[:1000]}


def revise_parser(frames: dict, previous: dict | None = None) -> dict:
    previous = previous or {}
    evaluations = [evaluate_policy(frames, policy) for policy in POLICIES]
    baseline = evaluations[0]
    eligible = [item for item in evaluations if item["policy"] != "baseline"
                and item["total"] >= 20
                and item["accuracy"] >= baseline["accuracy"] + 0.02
                and item["parse_coverage"] >= baseline["parse_coverage"] - 0.02]
    selected = max(eligible, key=lambda item: (item["accuracy"],
                                                item["prediction_coverage"],
                                                item["parse_coverage"]), default=baseline)
    revisions = list(previous.get("revisions", []))
    before = previous.get("selected_policy")
    if before and before != selected["policy"]:
        revisions.append({"before": before, "after": selected["policy"],
                          "reason": "unseen sequence prediction changed parser ranking",
                          "observations": len(frames)})
    failures = Counter()
    for trial in selected["trials"]:
        if not trial["correct"]:
            failures["next_event_prediction_mismatch"] += 1
    selected_public = {key: value for key, value in selected.items() if key != "trials"}
    return {"version": 38, "selected_policy": selected["policy"],
            "selection_status": ("predictive_revision_accepted" if eligible else
                                 "baseline_retained_no_candidate_improved"),
            "selected_evaluation": selected_public,
            "evaluations": [{key: value for key, value in item.items() if key != "trials"}
                            for item in evaluations],
            "counterexamples": selected["trials"][-500:],
            "failure_causes": dict(failures), "revisions": revisions[-100:],
            "warning": "parser policy changes only after holdout improvement; parsed events remain observations"}
