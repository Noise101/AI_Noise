#!/usr/bin/env python3
"""Remove redundant global-curiosity copies while preserving seed-specific learning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def local_gap_ids(state: dict) -> set[str]:
    return {cycle.get("gap", {}).get("gap_id") for cycle in state.get("cycles", [])
            if cycle.get("gap", {}).get("gap_id")}


def compact_state(state: dict) -> tuple[dict, dict]:
    before = state.get("curiosity_ledger", {})
    keep = local_gap_ids(state)
    state["curiosity_ledger"] = {
        gap_id: {key: value for key, value in entry.items() if key != "seed_encounters"}
        for gap_id, entry in before.items() if gap_id in keep
    }
    return state, {"curiosity_before": len(before), "curiosity_after": len(state["curiosity_ledger"]),
                   "cycles": len(state.get("cycles", [])),
                   "completed_gaps": len(state.get("completed_gap_ids", []))}


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


def compact_runtime(runtime: Path, apply: bool) -> dict:
    records = []
    paths = sorted(runtime.rglob("controller-state.json"))
    reports = sorted(runtime.rglob("latest-report.json"))
    for path in paths:
        before_size = path.stat().st_size
        value = json.loads(path.read_text(encoding="utf-8"))
        value, counts = compact_state(value)
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
