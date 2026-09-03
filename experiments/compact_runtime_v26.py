#!/usr/bin/env python3
"""Remove redundant global-curiosity copies while preserving seed-specific learning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from curiosity_drive_v23 import curiosity_pressure


REPORT_COMPACTION_VERSION = 1


def local_gap_ids(state: dict) -> set[str]:
    return {cycle.get("gap", {}).get("gap_id") for cycle in state.get("cycles", [])
            if cycle.get("gap", {}).get("gap_id")}


def compact_state(state: dict) -> tuple[dict, dict]:
    before = state.get("curiosity_ledger", {})
    rebuilt = {}
    for index, cycle in enumerate(state.get("cycles", [])):
        gap = cycle.get("gap", {})
        gap_id = gap.get("gap_id")
        if not gap_id:
            continue
        observations = max(1, int(gap.get("observations", 1)))
        entry = rebuilt.setdefault(gap_id, {
            "layer": gap.get("layer"), "query": gap.get("query"),
            "first_seen_cycle": index, "last_seen_cycle": index,
            "encounters": 0, "contexts_seen": 1, "status": "wanting_to_know",
            "resolution": None,
        })
        entry["encounters"] = max(entry["encounters"], observations)
        entry["last_seen_cycle"] = index
        entry["last_attempt_encounters"] = observations
        if cycle.get("grounded"):
            old = before.get(gap_id, {})
            entry.update({"status": "satisfied_for_now", "resolution": old.get("resolution"),
                          "resolved_cycle": index, "pressure": 0.0})
    for entry in rebuilt.values():
        if entry["status"] != "satisfied_for_now":
            entry["pressure"] = curiosity_pressure(entry, 1.0, len(state.get("cycles", [])))
    state["curiosity_ledger"] = rebuilt
    return state, {"curiosity_before": len(before), "curiosity_after": len(state["curiosity_ledger"]),
                   "cycles": len(state.get("cycles", [])),
                   "completed_gaps": len(state.get("completed_gap_ids", []))}


def rebuild_global_curiosity(states: list[dict], cycle: int) -> dict[str, dict]:
    rebuilt = {}
    for state in states:
        for gap_id, local in state.get("curiosity_ledger", {}).items():
            entry = rebuilt.setdefault(gap_id, {
                "layer": local.get("layer"), "query": local.get("query"),
                "first_seen_cycle": 0, "last_seen_cycle": cycle,
                "encounters": 0, "contexts_seen": 0, "status": "wanting_to_know",
                "resolution": None,
            })
            entry["encounters"] += local.get("encounters", 0)
            entry["contexts_seen"] += 1
            if local.get("status") == "satisfied_for_now":
                entry["status"] = "satisfied_for_now"
                entry["resolution"] = local.get("resolution")
    for entry in rebuilt.values():
        entry["pressure"] = (0.0 if entry["status"] == "satisfied_for_now"
                             else curiosity_pressure(entry, 1.0, cycle))
    return rebuilt


def compact_curriculum(curriculum: dict) -> tuple[dict, dict]:
    ledger = curriculum.get("curiosity_ledger", {})
    removed_seed_links = 0
    for entry in ledger.values():
        removed_seed_links += len(entry.get("seed_encounters", {}))
        entry.pop("seed_encounters", None)
    curriculum["curiosity_merge_seed"] = None
    curriculum["curiosity_merge_offsets"] = {}
    return curriculum, {"global_curiosity": len(ledger), "removed_seed_links": removed_seed_links,
                        "completed_seeds": len(curriculum.get("completed_seeds", [])),
                        "mastery_history": len(curriculum.get("mastery_history", []))}


def atomic_json(path: Path, value: dict) -> int:
    temporary = path.with_suffix(path.suffix + ".compact.tmp")
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return len(payload.encode())


def compact_seed_report(report: dict) -> tuple[dict, dict]:
    """Keep seed evidence but remove reconstructible copies of global priors."""
    if report.get("storage_compaction", {}).get("version") == REPORT_COMPACTION_VERSION:
        return report, {"already_compacted": True}
    knowledge = report.get("knowledge", {})
    bootstrap = knowledge.get("bootstrap", {})
    lexicon = knowledge.get("lexicon", {})
    word_forms = lexicon.get("word_forms", {})
    phrase_candidates = lexicon.get("phrase_candidates", [])
    phrase_forms = {item.get("phrase") for item in phrase_candidates if item.get("phrase")}
    conversation_cues = lexicon.get("conversation_cues", {})
    compact_lexicon = {
        name: lexicon.get(name, default) for name, default in (
            ("sentences_seen", 0), ("character_inventory", 0), ("characters", {}),
            ("word_forms", {}), ("grounded_meanings", []), ("phrase_candidates", []),
            ("conversation_cues", {}), ("meaning_revisions", []),
        )
    }
    compact_lexicon["researched_meanings"] = {
        key: value for key, value in lexicon.get("researched_meanings", {}).items()
        if key in word_forms and isinstance(value, dict) and value.get("accepted_sense")
    }
    compact_lexicon["researched_phrase_meanings"] = {
        key: value for key, value in lexicon.get("researched_phrase_meanings", {}).items()
        if key in phrase_forms and isinstance(value, dict) and value.get("accepted_sense")
    }
    compact_lexicon["researched_conversation_acts"] = {
        key: value for key, value in lexicon.get("researched_conversation_acts", {}).items()
        if key in conversation_cues and isinstance(value, dict) and value.get("accepted_sense")
    }
    knowledge["bootstrap"] = {
        name: bootstrap.get(name) for name in (
            "seed_concept", "first_generated_query", "sources",
            "next_self_generated_goal", "generated_at") if name in bootstrap
    }
    knowledge["lexicon"] = compact_lexicon
    report["knowledge"] = knowledge
    state = report.get("state", {})
    report["state"] = {name: state.get(name) for name in (
        "seed", "stop_reason", "created_at", "updated_at") if name in state}
    report["storage_compaction"] = {
        "version": REPORT_COMPACTION_VERSION,
        "kind": "historical_seed_evidence",
        "reconstructible_global_priors_removed": True,
    }
    return report, {"already_compacted": False}


def compact_historical_seed_reports(runtime: Path, current_seed: str | None = None,
                                    max_files: int = 250, apply: bool = True) -> dict:
    """Incrementally compact inactive reports without pausing normal learning."""
    processed = reclaimed = 0
    for path in sorted((runtime / "seeds").glob("*/latest-report.json")):
        if processed >= max_files:
            break
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if value.get("state", {}).get("seed") == current_seed:
            continue
        compacted, detail = compact_seed_report(value)
        if detail["already_compacted"]:
            continue
        before = path.stat().st_size
        after = atomic_json(path, compacted) if apply else len(
            (json.dumps(compacted, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        processed += 1
        reclaimed += max(0, before - after)
    return {"processed_reports": processed, "bytes_reclaimed": reclaimed,
            "batch_limit": max_files, "status": "applied" if apply else "dry_run"}


def compact_runtime(runtime: Path, apply: bool) -> dict:
    records = []
    compacted_states = []
    paths = sorted(runtime.rglob("controller-state.json"))
    reports = sorted(runtime.rglob("latest-report.json"))
    for path in paths:
        before_size = path.stat().st_size
        value = json.loads(path.read_text(encoding="utf-8"))
        value, counts = compact_state(value)
        compacted_states.append(value)
        after_size = atomic_json(path, value) if apply else before_size
        records.append({"path": str(path.relative_to(runtime)), "kind": "state",
                        "before_bytes": before_size, "after_bytes": after_size, **counts})
    for path in reports:
        before_size = path.stat().st_size
        value = json.loads(path.read_text(encoding="utf-8"))
        state = value.get("state", {})
        value["state"], counts = compact_state(state)
        after_size = atomic_json(path, value) if apply else before_size
        records.append({"path": str(path.relative_to(runtime)), "kind": "report",
                        "before_bytes": before_size, "after_bytes": after_size, **counts})
    curriculum_path = runtime / "curriculum-state.json"
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    before_size = curriculum_path.stat().st_size
    curriculum, curriculum_counts = compact_curriculum(curriculum)
    curriculum["curiosity_ledger"] = rebuild_global_curiosity(
        compacted_states, len(curriculum.get("mastery_history", [])))
    current_seed = curriculum.get("current_seed")
    current_state = next((state for state in compacted_states if state.get("seed") == current_seed), {})
    curriculum["curiosity_merge_seed"] = current_seed
    curriculum["curiosity_merge_offsets"] = {
        gap_id: entry.get("encounters", 0)
        for gap_id, entry in current_state.get("curiosity_ledger", {}).items()
    }
    after_size = atomic_json(curriculum_path, curriculum) if apply else before_size
    records.append({"path": "curriculum-state.json", "kind": "curriculum",
                    "before_bytes": before_size, "after_bytes": after_size, **curriculum_counts})
    summary = {"status": "applied" if apply else "dry_run", "files": len(records),
               "before_bytes": sum(item["before_bytes"] for item in records),
               "after_bytes": sum(item["after_bytes"] for item in records),
               "bytes_reclaimed": sum(item["before_bytes"] - item["after_bytes"] for item in records),
               "seed_states": len(paths), "reports": len(reports),
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if apply:
        atomic_json(runtime / "compaction-manifest.json", {"summary": summary, "records": records})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=Path(__file__).resolve().parent.parent / ".local")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(compact_runtime(args.runtime, args.apply), ensure_ascii=False))


if __name__ == "__main__":
    main()
