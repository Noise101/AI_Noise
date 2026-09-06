#!/usr/bin/env python3
"""One continuous, multi-dimensional capability dashboard.

Replaces "is there a plateau: yes/no" with per-dimension continuous signals so
the acquisition loop has a target and a human can see fine-grained movement:

  * per prediction task: accuracy, lift, significance, coverage, trend, and
    *learning efficiency* -- the slope of lift against cumulative training data
    (a flat slope over a big data increase is the real plateau signal).
  * held-out sequence-model perplexity (bits/char) once sequence_model_v1 runs.
  * extraction health: coreference resolutions, unresolved-pronoun rate,
    proposition yield, corpus growth.
  * a handful of boolean capability gates.

Pure aggregation over the plain module reports; no pretrained model.
"""

from __future__ import annotations


def _slope(points: list[tuple[float, float]]) -> float:
    """Least-squares slope of y against x; 0.0 when undefined."""
    n = len(points)
    if n < 2:
        return 0.0
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0
    return (n * sxy - sx * sy) / denom


def _task_dimension(event_structure: dict, task: str) -> dict:
    evaluation = next((item.get("selection", {})
                       for item in event_structure.get("evaluations", [])
                       if item.get("task") == task), {})
    total = evaluation.get("total", 0)
    curve = [p for p in event_structure.get("learning_curve", []) if p.get("task") == task]
    tail = curve[-20:]
    efficiency = _slope([(p.get("training_events", 0), p.get("lift", 0)) for p in tail])
    selected = event_structure.get("selected") or {}
    confirmed = selected.get("task") == task
    return {
        "task": task,
        "accuracy": round(evaluation.get("correct", 0) / total, 4) if total else 0.0,
        "baseline_accuracy": round(evaluation.get("baseline_correct", 0) / total, 4) if total else 0.0,
        "lift": evaluation.get("lift", 0),
        "coverage": evaluation.get("coverage", 0.0),
        "one_sided_sign_p": evaluation.get("one_sided_sign_p", 1.0),
        "total": total,
        "confirmed_on_final_split": confirmed,
        "final_lift": (selected.get("final") or {}).get("lift", 0) if confirmed else 0,
        "learning_curve_points": len(curve),
        "lift_per_1k_training_events": round(efficiency * 1000, 3),
        "trend": event_structure.get("learning_curve_trend", "insufficient_data"),
    }


def build_capability_report(event_structure: dict, verified_summary: dict,
                            experience_revision: dict, previous: dict | None = None,
                            sequence_model: dict | None = None) -> dict:
    previous = previous or {}
    sequence_model = sequence_model or {}
    dimensions = {task: _task_dimension(event_structure, task)
                  for task in ("verb_cloze", "event_plausibility")}

    accepted = verified_summary.get("accepted_sentences", 0)
    prior_accepted = previous.get("extraction", {}).get("accepted_sentences", accepted)
    extraction = {
        "accepted_sentences": accepted,
        "accepted_sentences_delta": accepted - prior_accepted,
        "unique_events": verified_summary.get("unique_events", 0),
        "coreference_resolutions": verified_summary.get("coreference_resolutions", 0),
        "unresolved_pronoun_subjects": verified_summary.get(
            "quarantine_reasons", {}).get("unresolved_pronoun_subject", 0),
        "propositions": verified_summary.get("propositions", 0),
        "entities_with_properties": verified_summary.get("entities_with_properties", 0),
    }

    revision_summary = experience_revision.get("summary", experience_revision)
    revision_eval = revision_summary.get("evaluation", {})
    revision = {
        "reusable_rules": revision_summary.get("reusable_rules", 0),
        "material_lift": (revision_eval.get("correct", 0)
                          - revision_eval.get("baseline_correct", 0)) >= max(
                              5, round(revision_eval.get("total", 0) * 0.05)),
        "evaluation": revision_eval,
    }

    sequence = {
        "held_out_bits_per_char": sequence_model.get("held_out_bits_per_char"),
        "baseline_bits_per_char": sequence_model.get("baseline_bits_per_char"),
        "improvement_bits": sequence_model.get("improvement_bits"),
        "improvement_z": sequence_model.get("improvement_z"),
        "improvement_p_one_sided": sequence_model.get("improvement_p_one_sided"),
        "beats_char_baseline": bool(sequence_model.get("beats_char_baseline")),
        "trend": sequence_model.get("perplexity_trend", "not_yet_measured"),
        "generative": bool(sequence_model.get("can_sample")),
    }

    gates = {
        "a_task_beats_baseline_on_frozen_final":
            any(d["confirmed_on_final_split"] for d in dimensions.values()),
        "some_task_learning_curve_is_rising":
            any(d["lift_per_1k_training_events"] > 0 and d["trend"] == "improving"
                for d in dimensions.values()),
        "reusable_rules_beat_baseline": revision["material_lift"] and revision["reusable_rules"] > 0,
        "sequence_model_beats_char_baseline": sequence["beats_char_baseline"],
        "extraction_is_growing": extraction["accepted_sentences_delta"] >= 0,
    }

    return {
        "version": 1,
        "dimensions": dimensions,
        "extraction": extraction,
        "experience_revision": revision,
        "sequence_model": sequence,
        "capability_gates": gates,
        "capability_gates_passed": sum(bool(v) for v in gates.values()),
        "capability_gates_total": len(gates),
        "headline": _headline(dimensions, gates),
    }


def _headline(dimensions: dict, gates: dict) -> str:
    confirmed = [d["task"] for d in dimensions.values() if d["confirmed_on_final_split"]]
    if confirmed:
        return f"confirmed on frozen final split: {', '.join(confirmed)}"
    rising = [d["task"] for d in dimensions.values() if d["trend"] == "improving"]
    if rising:
        return f"no confirmed capability yet; rising: {', '.join(rising)}"
    return "no confirmed capability; no task trending up"
