#!/usr/bin/env python3
"""Observation-only vessels for future human-science learning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SPEECH_ACTIONS = {"said", "asked", "replied", "answered", "cried", "told", "spoke"}


def empty_scaffold() -> dict:
    return {"version": 34, "mode": "observation_only", "observed_seeds": [], "frames": {},
            "field_vessels": {
                "psychology": {"behavioral_observations": True,
                               "mental_state_hypotheses": "reserved_empty"},
                "linguistics": {"forms_and_utterances": True,
                                "structural_and_pragmatic_hypotheses": "reserved_empty"},
                "philosophy": {"claims_and_arguments": "reserved_empty",
                               "normative_classification": "reserved_empty"},
                "sociology": {"actors_and_relations": "reserved_empty",
                              "group_and_institution_hypotheses": "reserved_empty"},
            }, "summary": {}}


def frame_id(seed: str, url: str, position: int, sentence: str) -> str:
    return hashlib.sha256(f"{seed}\n{url}\n{position}\n{sentence}".encode()).hexdigest()[:20]


def empty_lenses() -> dict:
    return {
        "psychology": {"mental_state_hypotheses": [], "alternative_explanations": []},
        "linguistics": {"form_patterns": [], "pragmatic_functions": []},
        "philosophy": {"claims": [], "premises": [], "inferences": [],
                       "objections": [], "normative_status": "unclassified"},
        "sociology": {"social_roles": [], "actor_relations": [],
                      "group_contexts": [], "institutional_contexts": []},
    }


def summarize(scaffold: dict) -> dict:
    frames = list(scaffold.get("frames", {}).values())
    return {"mode": scaffold.get("mode", "observation_only"),
            "observed_curricula": len(scaffold.get("observed_seeds", [])),
            "observation_frames": len(frames),
            "reported_speech_frames": sum(bool(item.get("utterance")) for item in frames),
            "interpretations_committed": sum(len(item.get("interpretations", [])) for item in frames),
            "hypotheses_committed": sum(len(item.get("hypotheses", [])) for item in frames),
            "decision_influence": False}


def observe_report(scaffold: dict, seed: str, report: dict) -> bool:
    if seed in set(scaffold.get("observed_seeds", [])):
        return False
    frames = scaffold.setdefault("frames", {})
    for source in report.get("knowledge", {}).get("bootstrap", {}).get("sources", []):
        url = source.get("url", "")
        previous_id = None
        for position, item in enumerate(source.get("event_extraction_audit", [])):
            if not item.get("accepted") or not item.get("event"):
                previous_id = None
                continue
            sentence, event = item.get("sentence", ""), item["event"]
            identity = frame_id(seed, url, position, sentence)
            subject, action, *_ = (event.split("|", 2) + ["", ""])[:3]
            utterance = None
            if action in SPEECH_ACTIONS or '"' in sentence or "“" in sentence:
                utterance = {"speaker_surface": subject or None, "textual_evidence": sentence,
                             "propositional_content": None,
                             "reason_unparsed": "content not safely separable yet"}
            frames[identity] = {
                "frame_id": identity, "seed": seed,
                "observation": {"event": event, "sentence": sentence,
                                "source_url": url, "source_position": position,
                                "extraction_quality": item.get("quality")},
                "utterance": utterance,
                "sequence": {"previous_frame": previous_id, "next_frame": None},
                "perspectives": [], "interpretations": [], "hypotheses": [],
                "counterexamples": [], "normative_claims": [],
                "applicability": {"scope": "single_textual_observation",
                                  "generalization_allowed": False},
                "confidence": {"kind": "event_extraction_confidence",
                               "value": item.get("quality", 0.0),
                               "not_a_claim_about_reality": True},
                "disciplinary_lenses": empty_lenses(),
            }
            if previous_id and previous_id in frames:
                frames[previous_id]["sequence"]["next_frame"] = identity
            previous_id = identity
    scaffold.setdefault("observed_seeds", []).append(seed)
    scaffold["summary"] = summarize(scaffold)
    return True


def rebuild_scaffold(runtime: Path, admitted_seeds: set[str]) -> dict:
    scaffold = empty_scaffold()
    paths = [runtime / "latest-report.json",
             *sorted((runtime / "seeds").glob("*/latest-report.json"))]
    reports: dict[str, dict] = {}
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        seed = report.get("state", {}).get("seed")
        if seed in admitted_seeds:
            reports[seed] = report
    for seed in sorted(reports):
        observe_report(scaffold, seed, reports[seed])
    scaffold["summary"] = summarize(scaffold)
    return scaffold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=Path(__file__).resolve().parent.parent /
                        ".local")
    args = parser.parse_args()
    memory = json.loads((args.runtime / "global-language-memory.json").read_text(encoding="utf-8"))
    scaffold = rebuild_scaffold(args.runtime, set(memory.get("merged_seeds", [])))
    output = args.runtime / "epistemic-observations.json"
    output.write_text(json.dumps(scaffold, ensure_ascii=False, separators=(",", ":")) + "\n",
                      encoding="utf-8")
    print(json.dumps(scaffold["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
