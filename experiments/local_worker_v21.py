#!/usr/bin/env python3
"""Run AI_Noise locally without spending Codex or remote-model usage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME = ROOT / ".local"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def run_cycle(seed: str, runtime: Path, steps: int, seconds: float, network: int) -> dict:
    state = runtime / "controller-state.json"
    report = runtime / "latest-report.json"
    command = [
        sys.executable, str(Path(__file__).with_name("autonomous_controller_v20.py")), seed,
        "--state", str(state), "--output", str(report), "--max-steps", str(steps),
        "--max-seconds", str(seconds), "--max-network", str(network), "--summary",
    ]
    completed = subprocess.run(command, cwd=Path(__file__).parent, capture_output=True,
                               text=True, timeout=max(10, seconds + 30))
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout).strip()[-1000:])
    return read_json(report)


def status_record(seed: str, runtime: Path, phase: str, rounds: int,
                  report: dict | None = None, error: str | None = None) -> dict:
    report = report or {}
    state = report.get("state", read_json(runtime / "controller-state.json"))
    return {
        "phase": phase,
        "seed": seed,
        "heartbeat": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rounds": rounds,
        "completed_gaps": len(state.get("completed_gap_ids", [])),
        "remaining_gaps": len(report.get("current_gaps", [])),
        "last_stop_reason": state.get("stop_reason"),
        "actual_network_requests": report.get("web_usage", {}).get("network_requests", 0),
        "error": error,
        "codex_or_remote_llm_calls": 0,
    }


def work(seed: str, runtime: Path, max_rounds: int, interval: float,
         steps: int, seconds: float, network: int) -> dict:
    runtime.mkdir(parents=True, exist_ok=True)
    status_path = runtime / "status.json"
    stop_path = runtime / "STOP"
    stop_path.unlink(missing_ok=True)
    latest = status_record(seed, runtime, "starting", 0)
    write_json(status_path, latest)
    for round_number in range(1, max_rounds + 1):
        if stop_path.exists():
            latest = status_record(seed, runtime, "stopped_by_user", round_number - 1)
            write_json(status_path, latest)
            return latest
        write_json(status_path, status_record(seed, runtime, "learning", round_number - 1))
        try:
            report = run_cycle(seed, runtime, steps, seconds, network)
        except Exception as error:
            latest = status_record(seed, runtime, "error", round_number, error=str(error))
            write_json(status_path, latest)
            return latest
        reason = report.get("state", {}).get("stop_reason")
        terminal = reason in {"no_unresolved_executable_gap", "network_budget_exhausted"}
        latest = status_record(seed, runtime, "idle" if terminal else "between_rounds",
                               round_number, report)
        write_json(status_path, latest)
        if terminal:
            return latest
        if interval > 0 and round_number < max_rounds:
            time.sleep(interval)
    latest["phase"] = "round_budget_exhausted"
    write_json(status_path, latest)
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run local learning rounds")
    run_parser.add_argument("seed", nargs="?", default="fox grapes")
    run_parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    run_parser.add_argument("--max-rounds", type=int, default=20)
    run_parser.add_argument("--interval", type=float, default=2)
    run_parser.add_argument("--steps", type=int, default=1)
    run_parser.add_argument("--seconds", type=float, default=60)
    run_parser.add_argument("--network", type=int, default=8)
    status_parser = subparsers.add_parser("status", help="show the last local heartbeat")
    status_parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    stop_parser = subparsers.add_parser("stop", help="request a safe stop between cycles")
    stop_parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    args = parser.parse_args()

    if args.command == "status":
        print(json.dumps(read_json(args.runtime / "status.json"), ensure_ascii=False, indent=2))
        return
    if args.command == "stop":
        args.runtime.mkdir(parents=True, exist_ok=True)
        (args.runtime / "STOP").touch()
        print("stop requested")
        return
    result = work(args.seed, args.runtime, args.max_rounds, args.interval,
                  args.steps, args.seconds, args.network)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
