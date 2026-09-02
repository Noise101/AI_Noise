#!/usr/bin/env python3
"""Stage-five abstraction, transfer and representation learning in bounded worlds."""

from __future__ import annotations

import hashlib
import itertools
import math
import time
from collections import Counter


COMPETENCIES = ("feature_comparison", "concept_formation", "concept_hierarchy",
                "relation_abstraction", "event_abstraction", "causal_transfer",
                "analogy", "structural_association", "self_revision",
                "representation_selection", "integrated_world_model")


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sid(track: str, scenario: dict) -> str:
    return track + ":" + ":".join(f"{k}={scenario[k]}" for k in sorted(scenario))


def _heldout(track: str, scenario: dict) -> bool:
    return hashlib.sha256(("stage5-holdout:" + _sid(track, scenario)).encode()).digest()[0] % 4 == 0


def _feature_scenarios() -> list[dict]:
    return [{"task": task, "color_same": c, "shape_same": s, "use_same": u}
            for task, c, s, u in itertools.product(("paint", "fit", "function"),
                                                    (False, True), (False, True), (False, True))]


def _feature(model: str, s: dict) -> str:
    relevant = {"paint": "color_same", "fit": "shape_same", "function": "use_same"}[s["task"]]
    if model == "task_relevant_feature":
        return "same" if s[relevant] else "different"
    if model == "all_features":
        return "same" if all(s[x] for x in ("color_same", "shape_same", "use_same")) else "different"
    if model == "color_only":
        return "same" if s["color_same"] else "different"
    if model == "shape_only":
        return "same" if s["shape_same"] else "different"
    return "different"


def _concept_scenarios() -> list[dict]:
    return [{"has_feathers": f, "lays_eggs": e, "flies": flies, "has_fur": fur}
            for f, e, flies, fur in itertools.product((False, True), repeat=4)]


def _concept(model: str, s: dict) -> str:
    if model == "defining_vs_typical":
        if s["has_feathers"] and s["lays_eggs"]:
            return "bird"
        if s["has_fur"]:
            return "mammal"
        return "unknown"
    if model == "flies_means_bird":
        return "bird" if s["flies"] else "not_bird"
    if model == "eggs_mean_bird":
        return "bird" if s["lays_eggs"] else "not_bird"
    if model == "fur_means_mammal":
        return "mammal" if s["has_fur"] else "unknown"
    return "unknown"


def _hierarchy_scenarios() -> list[dict]:
    return [{"kind": kind, "property": prop, "exception": exception}
            for kind, prop, exception in itertools.product(("lemon", "sparrow", "penguin", "stone"),
                                                            ("is_object", "can_fly", "is_food"),
                                                            (False, True))]


def _hierarchy(model: str, s: dict) -> str:
    parent = {"lemon": "fruit", "sparrow": "bird", "penguin": "bird", "stone": "object"}[s["kind"]]
    if model == "typed_inheritance_with_exceptions":
        if s["property"] == "is_object":
            return "yes"
        if s["property"] == "can_fly":
            return "no" if s["kind"] in {"lemon", "stone", "penguin"} or s["exception"] else "yes"
        return "yes" if parent == "fruit" and not s["exception"] else "no"
    if model == "copy_all_parent_traits":
        return "yes" if parent in {"fruit", "bird"} else "no"
    if model == "individual_only":
        return "unknown"
    if model == "everything_is_object_only":
        return "yes" if s["property"] == "is_object" else "no"
    return "no"


def _relation_scenarios() -> list[dict]:
    return [{"relation": relation, "query_direction": direction, "renamed": renamed}
            for relation, direction, renamed in itertools.product(("inside", "above", "before"),
                                                                   ("forward", "inverse"),
                                                                   (False, True))]


def _relation(model: str, s: dict) -> str:
    inverse = {"inside": "contains", "above": "below", "before": "after"}
    if model == "directed_relation":
        return s["relation"] if s["query_direction"] == "forward" else inverse[s["relation"]]
    if model == "symmetric_relation":
        return s["relation"]
    if model == "surface_names" and s["renamed"]:
        return "unknown"
    if model == "inside_only":
        return "inside" if s["relation"] == "inside" else "unknown"
    return "unknown"


def _event_scenarios() -> list[dict]:
    return [{"verb": verb, "success": success, "renamed_roles": renamed}
            for verb, success, renamed in itertools.product(("take", "obtain", "grab"),
                                                             (False, True), (False, True))]


def _event(model: str, s: dict) -> str:
    if model == "role_event_schema":
        return "agent>desire>attempt>" + ("success" if s["success"] else "failure")
    if model == "verb_identity":
        return s["verb"] + (">success" if s["success"] else ">failure")
    if model == "surface_roles" and s["renamed_roles"]:
        return "unknown"
    if model == "result_only":
        return "success" if s["success"] else "failure"
    return "attempt"


def _causal_scenarios() -> list[dict]:
    return [{"domain": domain, "input": value, "threshold": threshold, "blocked": blocked}
            for domain, value, threshold, blocked in itertools.product(
                ("push", "heat", "flow", "novel_signal"), (1, 3), (2, 3), (False, True))]


def _causal(model: str, s: dict) -> str:
    if model == "threshold_state_change":
        return "changes" if s["input"] >= s["threshold"] and not s["blocked"] else "stable"
    if model == "domain_surface":
        return "changes" if s["domain"] in {"push", "heat"} and s["input"] == 3 else "stable"
    if model == "high_input_always":
        return "changes" if s["input"] == 3 else "stable"
    if model == "correlation_only":
        return "changes" if s["domain"] == "push" else "stable"
    return "stable"


def _analogy_scenarios() -> list[dict]:
    return [{"source": source, "target": target, "valid_condition": valid}
            for source, target, valid in itertools.product(("key", "password", "permit"),
                                                            ("door", "terminal", "zone"),
                                                            (False, True))]


def _analogy(model: str, s: dict) -> str:
    matching = {"key": "door", "password": "terminal", "permit": "zone"}
    if model == "role_mapping":
        return "grants_access" if matching[s["source"]] == s["target"] and s["valid_condition"] else "no_access"
    if model == "word_similarity":
        return "grants_access" if s["source"][0] == s["target"][0] else "no_access"
    if model == "all_credentials_work":
        return "grants_access"
    if model == "condition_only":
        return "grants_access" if s["valid_condition"] else "no_access"
    return "no_access"


def _association_scenarios() -> list[dict]:
    return [{"concept": concept, "requested_relation": relation, "surface_distractor": distractor}
            for concept, relation, distractor in itertools.product(("lemon", "nest", "hammer", "novel_fruit"),
                                                                    ("part", "location", "use"),
                                                                    (False, True))]


def _association(model: str, s: dict) -> str:
    typed = {
        "lemon": {"part": "peel", "location": "tree", "use": "drink"},
        "novel_fruit": {"part": "skin", "location": "plant", "use": "food"},
        "nest": {"part": "twig", "location": "tree", "use": "shelter"},
        "hammer": {"part": "handle", "location": "toolbox", "use": "strike"},
    }
    if model == "typed_relation_graph":
        return typed[s["concept"]][s["requested_relation"]]
    if model == "frequency_baseline":
        return "tree"
    if model == "surface_distractor":
        return "yellow" if s["surface_distractor"] else "tree"
    if model == "concept_only":
        return s["concept"]
    return "unknown"


def _revision_scenarios() -> list[dict]:
    return [{"category": category, "typical": typical, "exception": exception}
            for category, typical, exception in itertools.product(("bird", "tool", "fruit"),
                                                                   (False, True), (False, True))]


def _revision(model: str, s: dict) -> str:
    if model == "scoped_rule_with_exception":
        return "applies" if s["typical"] and not s["exception"] else "does_not_apply"
    if model == "initial_overgeneralization":
        return "applies" if s["category"] in {"bird", "tool", "fruit"} else "does_not_apply"
    if model == "delete_rule_after_error":
        return "does_not_apply"
    if model == "typical_only":
        return "applies" if s["typical"] else "does_not_apply"
    return "applies"


REPRESENTATIONS = ("surface", "feature_set", "event_graph", "relation_graph")


def _representation_scenarios() -> list[dict]:
    return [{"task": task, "renamed": renamed, "distractors": distractors}
            for task, renamed, distractors in itertools.product(("classify", "event", "relation"),
                                                                 (False, True), (False, True))]


def _representation(model: str, s: dict) -> str:
    best = {"classify": "feature_set", "event": "event_graph", "relation": "relation_graph"}
    if model == "task_selected_representation":
        return best[s["task"]]
    if model in REPRESENTATIONS:
        return model
    return "surface"


def _world_model_scenarios() -> list[dict]:
    return [{"query": query, "has_counterexample": counter, "source_available": source}
            for query, counter, source in itertools.product(("evidence", "status", "revision", "cause"),
                                                             (False, True), (False, True))]


def _world_model(model: str, s: dict) -> str:
    if model == "linked_provenance_graph":
        if not s["source_available"]:
            return "uncertain_missing_source"
        if s["has_counterexample"]:
            return "revised_with_counterexample"
        return "traceable_" + s["query"]
    if model == "flat_facts":
        return "fact"
    if model == "latest_only":
        return "revised_with_counterexample" if s["has_counterexample"] else "fact"
    if model == "confidence_only":
        return "certain" if s["source_available"] else "uncertain_missing_source"
    return "unknown"


SPECS = {
    "feature_comparison": (_feature_scenarios, _feature,
                           ("task_relevant_feature", "all_features", "color_only", "shape_only", "always_different"),
                           "task_relevant_feature", "all_features"),
    "concept_formation": (_concept_scenarios, _concept,
                          ("defining_vs_typical", "flies_means_bird", "eggs_mean_bird", "fur_means_mammal", "unknown"),
                          "defining_vs_typical", "flies_means_bird"),
    "concept_hierarchy": (_hierarchy_scenarios, _hierarchy,
                          ("typed_inheritance_with_exceptions", "copy_all_parent_traits", "individual_only",
                           "everything_is_object_only", "always_no"),
                          "typed_inheritance_with_exceptions", "copy_all_parent_traits"),
    "relation_abstraction": (_relation_scenarios, _relation,
                             ("directed_relation", "symmetric_relation", "surface_names", "inside_only", "unknown"),
                             "directed_relation", "surface_names"),
    "event_abstraction": (_event_scenarios, _event,
                          ("role_event_schema", "verb_identity", "surface_roles", "result_only", "attempt_only"),
                          "role_event_schema", "verb_identity"),
    "causal_transfer": (_causal_scenarios, _causal,
                        ("threshold_state_change", "domain_surface", "high_input_always", "correlation_only", "always_stable"),
                        "threshold_state_change", "domain_surface"),
    "analogy": (_analogy_scenarios, _analogy,
                ("role_mapping", "word_similarity", "all_credentials_work", "condition_only", "never_access"),
                "role_mapping", "word_similarity"),
    "structural_association": (_association_scenarios, _association,
                               ("typed_relation_graph", "frequency_baseline", "surface_distractor", "concept_only", "unknown"),
                               "typed_relation_graph", "frequency_baseline"),
    "self_revision": (_revision_scenarios, _revision,
                      ("scoped_rule_with_exception", "initial_overgeneralization", "delete_rule_after_error",
                       "typical_only", "always_applies"),
                      "scoped_rule_with_exception", "initial_overgeneralization"),
    "representation_selection": (_representation_scenarios, _representation,
                                 ("task_selected_representation", "surface", "feature_set", "event_graph", "relation_graph"),
                                 "task_selected_representation", "surface"),
    "integrated_world_model": (_world_model_scenarios, _world_model,
                               ("linked_provenance_graph", "flat_facts", "latest_only", "confidence_only", "unknown"),
                               "linked_provenance_graph", "flat_facts"),
}

MIN_OBSERVATIONS = {name: 4 for name in COMPETENCIES}


def empty_abstraction_memory() -> dict:
    return {"version": 46, "stage": 5,
            "tracks": {name: {"candidates": list(SPECS[name][2]), "observations": [],
                              "revisions": []} for name in COMPETENCIES},
            "error_memory": [], "summary": {}, "remote_llm_calls": 0,
            "world_graph": {"nodes": {}, "evidence_links": [], "revision_links": []}}


def _majority(candidates: list[str], rule, scenario: dict) -> str:
    votes = Counter(rule(model, scenario) for model in candidates)
    return sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]


def _choose(track: str, state: dict) -> dict | None:
    generator, rule, _, _, _ = SPECS[track]
    tried = {x["scenario_id"] for x in state["observations"]}
    choices = []
    for scenario in generator():
        sid = _sid(track, scenario)
        if sid in tried or _heldout(track, scenario):
            continue
        disagreement = len({rule(model, scenario) for model in state["candidates"]})
        choices.append((disagreement, hashlib.sha256(sid.encode()).hexdigest(), scenario))
    return max(choices)[2] if choices else None


def _learn_one(memory: dict, track: str) -> bool:
    state = memory["tracks"][track]
    if len(state["candidates"]) == 1 and len(state["observations"]) >= MIN_OBSERVATIONS[track]:
        return False
    _, rule, _, truth, _ = SPECS[track]
    scenario = _choose(track, state)
    if scenario is None:
        return False
    before = list(state["candidates"])
    prediction = _majority(before, rule, scenario)
    actual = rule(truth, scenario)
    record = {"scenario_id": _sid(track, scenario), "context": scenario,
              "prediction": prediction, "observed": actual,
              "prediction_error": prediction != actual, "at": _stamp()}
    state["observations"].append(record)
    state["candidates"] = [model for model in before if rule(model, scenario) == actual]
    if before != state["candidates"]:
        state["revisions"].append({"before": before, "after": state["candidates"],
                                   "evidence": record["scenario_id"]})
    if record["prediction_error"]:
        memory["error_memory"].append({"competency": track, **record})
    node = memory["world_graph"]["nodes"].setdefault(track, {"kind": "learned_structure"})
    node["selected_model"] = state["candidates"][0] if len(state["candidates"]) == 1 else None
    memory["world_graph"]["evidence_links"].append({"from": record["scenario_id"], "to": track})
    if before != state["candidates"]:
        memory["world_graph"]["revision_links"].append({"evidence": record["scenario_id"], "rule": track})
    return True


def _evaluate_model(track: str, model: str | None) -> tuple[int, int]:
    generator, rule, _, truth, _ = SPECS[track]
    correct = total = 0
    for scenario in generator():
        if not _heldout(track, scenario):
            continue
        correct += model is not None and rule(model, scenario) == rule(truth, scenario)
        total += 1
    return correct, total


def _evaluate(track: str, state: dict) -> dict:
    _, _, _, _, baseline = SPECS[track]
    selected = state["candidates"][0] if len(state["candidates"]) == 1 else None
    correct, total = _evaluate_model(track, selected)
    baseline_correct, _ = _evaluate_model(track, baseline)
    return {"correct": correct, "total": total,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "baseline_correct": baseline_correct,
            "baseline_accuracy": round(baseline_correct / total, 4) if total else 0.0,
            "beats_baseline": correct > baseline_correct,
            "hypotheses_remaining": len(state["candidates"]), "selected_model": selected}


def learn_abstractions(memory: dict, steps: int = 12) -> dict:
    if memory.get("version") != 46:
        memory.clear()
        memory.update(empty_abstraction_memory())
    for _ in range(max(0, steps)):
        if not any(_learn_one(memory, track) for track in COMPETENCIES):
            break
    evaluations = {track: _evaluate(track, memory["tracks"][track]) for track in COMPETENCIES}
    passed = {track: result["accuracy"] == 1.0 and result["hypotheses_remaining"] == 1
              for track, result in evaluations.items()}
    causal_gate = evaluations["causal_transfer"]["beats_baseline"]
    association_gate = evaluations["structural_association"]["beats_baseline"]
    revision = evaluations["self_revision"]
    revision_gate = revision["correct"] > revision["baseline_correct"]
    representation_gate = (evaluations["representation_selection"]["selected_model"]
                           == "task_selected_representation"
                           and evaluations["representation_selection"]["beats_baseline"])
    reusable = sum(result["selected_model"] not in {None, "surface"}
                   and result["accuracy"] == 1.0 for result in evaluations.values())
    transfer_gate = reusable >= 5
    gates = {"causal_beats_baseline": causal_gate,
             "structural_association_beats_baseline": association_gate,
             "revision_improves_holdout": revision_gate,
             "non_surface_representation_selected": representation_gate,
             "abstract_rules_reused_on_unseen": transfer_gate}
    bounded_complete = all(passed.values()) and all(gates.values())
    score = sum(passed.values())
    status = ("stage_5_bounded_complete_open_transfer_pending" if bounded_complete else
              ("stage_5_integration_testing" if score >= 9 else
               ("stage_5_transfer_testing" if score >= 7 else
                ("stage_5_structure_learning" if score >= 4 else
                 ("stage_5_concept_learning" if score >= 2 else "stage_5_feature_learning")))))
    memory["summary"] = {
        "stage": 5, "status": status,
        "bounded_world_complete": bounded_complete, "open_transfer_complete": False,
        "competencies_passed": score,
        "competencies_total": len(COMPETENCIES), "competencies": evaluations,
        "required_gates": gates,
        "abstraction_experiments": sum(len(x["observations"]) for x in memory["tracks"].values()),
        "prediction_errors": len(memory["error_memory"]),
        "reusable_abstract_rules": reusable,
        "world_model": {"nodes": len(memory["world_graph"]["nodes"]),
                        "evidence_links": len(memory["world_graph"]["evidence_links"]),
                        "revision_links": len(memory["world_graph"]["revision_links"])},
        "remote_llm_calls": 0, "continues_autonomous_learning_after_completion": True,
        "limitations": ["completion is mastery of bounded abstraction and transfer worlds",
                        "transfer to open web text must still be measured separately"]}
    return memory["summary"]


def assess_open_transfer(memory: dict, representation: dict, association: dict,
                         causal: dict, revision: dict) -> dict:
    """Do not call stage 5 complete until abstractions improve real curriculum holdouts."""
    summary = memory.get("summary") or learn_abstractions(memory, 0)
    selected = representation.get("selected_evaluation", {})
    def material_lift(evaluation: dict) -> bool:
        total = evaluation.get("total", 0)
        required = max(5, math.ceil(total * 0.01))
        return evaluation.get("correct", 0) - evaluation.get("baseline_correct", 0) >= required

    representation_gate = (representation.get("selected_scheme") != "surface"
                           and material_lift(selected))
    association_eval = association.get("selected_evaluation", association.get("evaluation", {}))
    association_gate = material_lift(association_eval)
    causal_eval = causal.get("evaluation", {})
    causal_gate = material_lift(causal_eval)
    revision_summary = revision.get("summary", revision)
    revision_eval = revision_summary.get("evaluation", {})
    revision_gate = (revision_summary.get("reusable_rules", 0) > 0
                     and material_lift(revision_eval))
    open_gates = {"real_representation_beats_surface": representation_gate,
                  "real_association_beats_baseline": association_gate,
                  "real_causal_prediction_beats_baseline": causal_gate,
                  "real_reusable_rules_beat_baseline": revision_gate}
    summary["open_transfer_gates"] = open_gates
    summary["open_transfer_gate_policy"] = "at least 5 predictions and 1% of holdout above baseline"
    summary["open_transfer_complete"] = all(open_gates.values())
    if summary.get("bounded_world_complete") and summary["open_transfer_complete"]:
        summary["status"] = "stage_5_complete"
    elif summary.get("bounded_world_complete"):
        summary["status"] = "stage_5_bounded_complete_open_transfer_learning"
    memory["summary"] = summary
    return summary
