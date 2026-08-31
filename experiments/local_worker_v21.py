#!/usr/bin/env python3
"""Run AI_Noise locally without spending Codex or remote-model usage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

from web_cache import WEB_CACHE, NetworkBudgetExceeded
from curiosity_drive_v23 import curiosity_pressure
from mastery_drive_v24 import assess_language_mastery
from local_conversation_v25 import practice_once
from compact_runtime_v26 import compact_runtime
from global_memory_v27 import empty_memory, mastery_report, merge_report
from causal_experiment_v28 import CausalExperimentEngine


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME = ROOT / ".local"
WORD = re.compile(r"[A-Za-z]+")
TITLE_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "three",
              "hundred", "aesop", "s", "fables"}


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


def is_transient_error(error: Exception) -> bool:
    text = f"{type(error).__name__}: {error}".lower()
    markers = ("timeout", "timed out", "temporarily unavailable", "connection reset",
               "connection refused", "remote end closed", "http error 429", "http error 500",
               "http error 502", "http error 503", "http error 504", "name or service not known")
    return isinstance(error, (TimeoutError, ConnectionError, subprocess.TimeoutExpired)) or any(
        marker in text for marker in markers)


def wait_for_retry(stop_path: Path, seconds: float) -> bool:
    """Return False when a safe stop is requested during backoff."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop_path.exists():
            return False
        time.sleep(min(1, deadline - time.monotonic()))
    return True


def runtime_bytes(runtime: Path) -> int:
    return sum(path.stat().st_size for path in runtime.rglob("*") if path.is_file())


def enforce_storage_budget(runtime: Path, max_bytes: int) -> dict:
    before = runtime_bytes(runtime)
    compacted = None
    if before > max_bytes:
        compacted = compact_runtime(runtime, True)
    after = runtime_bytes(runtime)
    record = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "before_bytes": before, "after_bytes": after, "limit_bytes": max_bytes,
              "compacted": compacted is not None,
              "bytes_reclaimed": 0 if not compacted else compacted["bytes_reclaimed"]}
    write_json(runtime / "storage-status.json", record)
    return record


def seed_runtime(runtime: Path, seed: str) -> Path:
    legacy = read_json(runtime / "controller-state.json")
    if legacy.get("seed") == seed:
        return runtime
    identity = hashlib.sha256(seed.encode()).hexdigest()[:12]
    return runtime / "seeds" / identity


def run_cycle(seed: str, runtime: Path, steps: int, seconds: float, network: int,
              curiosity_priors: Path | None = None, global_memory: Path | None = None) -> dict:
    runtime = seed_runtime(runtime, seed)
    runtime.mkdir(parents=True, exist_ok=True)
    state = runtime / "controller-state.json"
    report = runtime / "latest-report.json"
    command = [
        sys.executable, str(Path(__file__).with_name("autonomous_controller_v20.py")), seed,
        "--state", str(state), "--output", str(report), "--max-steps", str(steps),
        "--max-seconds", str(seconds), "--max-network", str(network), "--summary",
    ]
    if curiosity_priors:
        command.extend(["--curiosity-priors", str(curiosity_priors)])
    if global_memory:
        command.extend(["--global-memory", str(global_memory)])
    completed = subprocess.run(command, cwd=Path(__file__).parent, capture_output=True,
                               text=True, timeout=max(10, seconds + 30))
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout).strip()[-1000:])
    return read_json(report)


def merge_curiosity(curriculum: dict, seed: str, report: dict, cycle: int) -> None:
    global_ledger = curriculum.setdefault("curiosity_ledger", {})
    if curriculum.get("curiosity_merge_seed") != seed:
        curriculum["curiosity_merge_seed"] = seed
        curriculum["curiosity_merge_offsets"] = {}
    offsets = curriculum.setdefault("curiosity_merge_offsets", {})
    local_ledger = report.get("state", {}).get("curiosity_ledger", {})
    for gap_id, local in local_ledger.items():
        entry = global_ledger.setdefault(gap_id, {
            "layer": local.get("layer"), "query": local.get("query"),
            "first_seen_cycle": cycle, "last_seen_cycle": cycle,
            "encounters": 0, "contexts_seen": 0,
            "status": "wanting_to_know", "resolution": None,
        })
        previous = offsets.get(gap_id, 0)
        observed = local.get("encounters", 0)
        if observed > previous:
            entry["encounters"] += observed - previous
            if gap_id not in offsets:
                entry["contexts_seen"] += 1
            offsets[gap_id] = observed
            entry["last_seen_cycle"] = cycle
        if local.get("status") == "satisfied_for_now":
            entry.update({"status": "satisfied_for_now", "resolution": local.get("resolution"),
                          "pressure": 0.0})
        elif entry.get("status") != "satisfied_for_now":
            entry["pressure"] = curiosity_pressure(entry, 1.0, cycle)


def _seed_from_title(title: str) -> str | None:
    leaf = urllib.parse.unquote(title).rsplit("/", 1)[-1].replace("_", " ")
    words = [word.lower() for word in WORD.findall(leaf)
             if word.lower() not in TITLE_STOP and len(word) > 2]
    unique = list(dict.fromkeys(words))
    return " ".join(unique[:3]) if len(unique) >= 2 else None


def discover_curriculum(report: dict, visited: set[str], network: int) -> list[dict]:
    """Generate next seeds from observed evidence links and learned concepts."""
    candidates: dict[str, dict] = {}
    sources = report.get("knowledge", {}).get("bootstrap", {}).get("sources", [])
    WEB_CACHE.set_network_budget(network)
    for source in sources:
        url = source.get("url", "")
        if "en.wikisource.org/wiki/" not in url:
            continue
        title = urllib.parse.unquote(url.split("/wiki/", 1)[1]).replace("_", " ")
        params = urllib.parse.urlencode({"action": "query", "prop": "links", "titles": title,
                                        "plnamespace": 0, "pllimit": 100,
                                        "format": "json", "formatversion": 2})
        try:
            data = WEB_CACHE.get_json("https://en.wikisource.org/w/api.php?" + params,
                                      "AI_Noise/0.22 (read-only autonomous curriculum)")
        except (NetworkBudgetExceeded, Exception):
            continue
        for page in data.get("query", {}).get("pages", []):
            for link in page.get("links", []):
                linked_title = link.get("title", "")
                seed = _seed_from_title(linked_title)
                if not seed or seed in visited:
                    continue
                same_collection = title.rsplit("/", 1)[0] in linked_title if "/" in title else False
                score = 3.0 if same_collection else 1.0
                candidates[seed] = {"seed": seed, "score": score,
                                    "reason": "unvisited story link found in read evidence",
                                    "parent_url": url, "linked_title": linked_title}
        if "/" in title:
            collection = title.rsplit("/", 1)[0] + "/"
            params = urllib.parse.urlencode({"action": "query", "list": "allpages",
                                             "apprefix": collection, "apnamespace": 0,
                                             "aplimit": 100, "format": "json", "formatversion": 2})
            try:
                shelf = WEB_CACHE.get_json("https://en.wikisource.org/w/api.php?" + params,
                                           "AI_Noise/0.24 (read-only observed-shelf curriculum)")
            except Exception:
                shelf = {}
            for page in shelf.get("query", {}).get("allpages", []):
                linked_title = page.get("title", "")
                seed = _seed_from_title(linked_title)
                if not seed or seed in visited:
                    continue
                candidates.setdefault(seed, {"seed": seed, "score": 2.5,
                                              "reason": "unvisited page in an observed story collection",
                                              "parent_url": url, "linked_title": linked_title})
    beliefs = report.get("knowledge", {}).get("concepts", {}).get("beliefs", [])
    for belief in beliefs:
        seed = f"{belief.get('subject', '')} {belief.get('object', '')}".strip().lower()
        if seed in visited or len(seed.split()) < 2:
            continue
        candidates.setdefault(seed, {"seed": seed, "score": 0.5,
                                     "reason": "unvisited concept pair from evidence ledger",
                                     "parent_url": (belief.get("citations") or [None])[0]})
    return sorted(candidates.values(), key=lambda item: (-item["score"], item["seed"]))


def rediscover_from_history(runtime: Path, visited: set[str], network: int) -> list[dict]:
    found: dict[str, dict] = {}
    reports = [runtime / "latest-report.json", *sorted((runtime / "seeds").glob("*/latest-report.json"))]
    for path in reports:
        report = read_json(path)
        if not report:
            continue
        for item in discover_curriculum(report, visited | set(found), network):
            found.setdefault(item["seed"], item)
        if len(found) >= 20:
            break
    return sorted(found.values(), key=lambda item: (-item["score"], item["seed"]))


def status_record(seed: str, runtime: Path, phase: str, rounds: int,
                  report: dict | None = None, error: str | None = None) -> dict:
    report = report or {}
    state = report.get("state", read_json(runtime / "controller-state.json"))
    mastery = report.get("mastery") or read_json(runtime / "mastery.json")
    causal_memory = read_json(runtime / "causal-memory.json")
    return {
        "phase": phase,
        "seed": seed,
        "heartbeat": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "rounds": rounds,
        "curricula_completed": len(read_json(runtime / "curriculum-state.json").get("completed_seeds", [])),
        "completed_gaps": len(state.get("completed_gap_ids", [])),
        "remaining_gaps": len(report.get("current_gaps", [])),
        "last_stop_reason": state.get("stop_reason"),
        "actual_network_requests": report.get("web_usage", {}).get("network_requests", 0),
        "error": error,
        "codex_or_remote_llm_calls": 0,
        "mastery": {"status": mastery.get("status"), "overall_score": mastery.get("overall_score"),
                    "weakest_dimension": mastery.get("weakest_dimension"),
                    "next_goal": mastery.get("next_mastery_goal")},
        "local_conversation": report.get("local_conversation"),
        "storage": read_json(runtime / "storage-status.json"),
        "global_memory": report.get("global_memory") or read_json(
            runtime / "global-language-memory.json").get("totals", {}),
        "causal_evaluation": report.get("causal_evaluation") or {
            key: causal_memory.get(key)
            for key in ("supported_hypotheses", "evaluation", "limitations")},
    }


def work(seed: str, runtime: Path, max_rounds: int, interval: float,
         steps: int, seconds: float, network: int, local_conversation: bool = True,
         max_runtime_mb: int = 1024) -> dict:
    runtime.mkdir(parents=True, exist_ok=True)
    status_path = runtime / "status.json"
    stop_path = runtime / "STOP"
    stop_path.unlink(missing_ok=True)
    curriculum_path = runtime / "curriculum-state.json"
    curriculum = read_json(curriculum_path) or {
        "initial_seed": seed, "current_seed": seed, "completed_seeds": [],
        "deferred_seeds": [], "frontier": [], "transitions": [], "curiosity_ledger": {},
    }
    for key, default in (("completed_seeds", []), ("deferred_seeds", []), ("frontier", []),
                         ("transitions", []), ("curiosity_ledger", {}), ("mastery_history", [])):
        curriculum.setdefault(key, default)
    curriculum.setdefault("conversation_practiced_seeds", [])
    seed = curriculum["current_seed"]
    write_json(curriculum_path, curriculum)
    enforce_storage_budget(runtime, max_runtime_mb * 1024 * 1024)
    latest = status_record(seed, runtime, "starting", 0)
    write_json(status_path, latest)
    round_number = 0
    consecutive_transient_errors = 0
    while max_rounds <= 0 or round_number < max_rounds:
        round_number += 1
        if round_number > 1 and round_number % 100 == 0:
            write_json(status_path, status_record(seed, runtime, "storage_check", round_number - 1))
            enforce_storage_budget(runtime, max_runtime_mb * 1024 * 1024)
        if stop_path.exists():
            latest = status_record(seed, runtime, "stopped_by_user", round_number - 1)
            write_json(status_path, latest)
            return latest
        write_json(status_path, status_record(seed, runtime, "learning", round_number - 1))
        try:
            report = run_cycle(seed, runtime, steps, seconds, network,
                               runtime / "curiosity-priors.json",
                               runtime / "global-language-memory.json")
        except Exception as error:
            if not is_transient_error(error):
                latest = status_record(seed, runtime, "error", round_number, error=str(error))
                write_json(status_path, latest)
                return latest
            consecutive_transient_errors += 1
            retry_seconds = min(30, max(2, 2 ** min(consecutive_transient_errors, 5)))
            latest = status_record(seed, runtime, "transient_error_wait", round_number,
                                   error=f"{type(error).__name__}: {error}")
            latest["retry_in_seconds"] = retry_seconds
            latest["consecutive_transient_errors"] = consecutive_transient_errors
            write_json(status_path, latest)
            if not wait_for_retry(stop_path, retry_seconds):
                latest["phase"] = "stopped_by_user"
                write_json(status_path, latest)
                return latest
            continue
        consecutive_transient_errors = 0
        reason = report.get("state", {}).get("stop_reason")
        memory_path = runtime / "global-language-memory.json"
        memory = read_json(memory_path) or empty_memory()
        new_global_experience = merge_report(memory, seed, report)
        write_json(memory_path, memory)
        if new_global_experience or not (runtime / "causal-memory.json").exists():
            # Legacy transitions predate extraction audits and remain quarantined from causal claims.
            causal_report = CausalExperimentEngine(
                memory.get("quality_event_transitions", {})).run()
            write_json(runtime / "causal-memory.json", causal_report)
        else:
            causal_report = read_json(runtime / "causal-memory.json")
        mastery = assess_language_mastery(mastery_report(memory))
        report["mastery"] = mastery
        report["global_memory"] = memory.get("totals", {})
        report["causal_evaluation"] = {
            "supported_hypotheses": causal_report.get("supported_hypotheses", 0),
            "evaluation": causal_report.get("evaluation", {}),
            "limitations": causal_report.get("limitations", []),
        }
        write_json(runtime / "mastery.json", mastery)
        curriculum["mastery_history"].append({"seed": seed, "round": round_number,
                                               "overall_score": mastery["overall_score"],
                                               "weakest_dimension": mastery["weakest_dimension"],
                                               "next_mastery_goal": mastery["next_mastery_goal"]})
        if local_conversation and seed not in curriculum["conversation_practiced_seeds"]:
            turn = practice_once(seed, mastery, curriculum["curiosity_ledger"])
            dialogue_path = runtime / "dialogue-ledger.json"
            dialogue = read_json(dialogue_path) or {"turns": []}
            dialogue["turns"].append(turn)
            write_json(dialogue_path, dialogue)
            curriculum["conversation_practiced_seeds"].append(seed)
            report["local_conversation"] = {"status": turn["status"],
                                             "turns_total": len(dialogue["turns"]),
                                             "evidence_score": 0.0}
        write_json(curriculum_path, curriculum)
        merge_curiosity(curriculum, seed, report, round_number)
        write_json(runtime / "curiosity-priors.json", {
            gap_id: {"pressure": item.get("pressure", 0.0), "status": item.get("status")}
            for gap_id, item in curriculum["curiosity_ledger"].items()
        })
        exhausted = reason in {"no_unresolved_executable_gap", "no_new_evidence_for_unresolved_gap"}
        if exhausted:
            bucket = "completed_seeds" if reason == "no_unresolved_executable_gap" else "deferred_seeds"
            if seed not in curriculum[bucket]:
                curriculum[bucket].append(seed)
            visited = set(curriculum["completed_seeds"]) | set(curriculum["deferred_seeds"])
            discovered = discover_curriculum(report, visited, network)
            known = {item["seed"] for item in curriculum["frontier"]}
            curriculum["frontier"].extend(item for item in discovered if item["seed"] not in known)
            curriculum["frontier"] = [item for item in curriculum["frontier"]
                                      if item["seed"] not in visited]
            if not curriculum["frontier"]:
                curriculum["frontier"].extend(rediscover_from_history(runtime, visited, network))
            if curriculum["frontier"]:
                selected = curriculum["frontier"].pop(0)
                curriculum["transitions"].append({"from": seed, "to": selected["seed"],
                                                   "reason": selected["reason"],
                                                   "parent_url": selected.get("parent_url")})
                seed = selected["seed"]
                curriculum["current_seed"] = seed
                write_json(curriculum_path, curriculum)
                latest = status_record(seed, runtime, "curriculum_transition", round_number, report)
                write_json(status_path, latest)
                continue
        resource_pause = reason == "network_budget_exhausted"
        phase = "resource_paused" if resource_pause else (
            "curriculum_exhausted" if exhausted else "between_rounds")
        latest = status_record(seed, runtime, phase,
                               round_number, report)
        write_json(status_path, latest)
        if exhausted:
            return latest
        if interval > 0 and (max_rounds <= 0 or round_number < max_rounds):
            time.sleep(interval)
    latest["phase"] = "round_budget_exhausted"
    write_json(status_path, latest)
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    def add_work_arguments(work_parser):
        work_parser.add_argument("seed", nargs="?", default="fox grapes")
        work_parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
        work_parser.add_argument("--max-rounds", type=int, default=0,
                                 help="0 keeps running until STOP or a genuine frontier boundary")
        work_parser.add_argument("--interval", type=float, default=2)
        work_parser.add_argument("--steps", type=int, default=1)
        work_parser.add_argument("--seconds", type=float, default=60)
        work_parser.add_argument("--network", type=int, default=8)
        work_parser.add_argument("--no-local-conversation", action="store_true")
        work_parser.add_argument("--max-runtime-mb", type=int, default=1024)
    run_parser = subparsers.add_parser("run", help="run in the foreground")
    add_work_arguments(run_parser)
    start_parser = subparsers.add_parser("start", help="start once and keep working in the background")
    add_work_arguments(start_parser)
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
    if args.command == "start":
        existing = read_json(args.runtime / "status.json")
        pid = existing.get("pid")
        if pid:
            try:
                os.kill(pid, 0)
                print(json.dumps({"status": "already_running", "pid": pid}, ensure_ascii=False))
                return
            except OSError:
                pass
        command = [sys.executable, str(Path(__file__).resolve()), "run", args.seed,
                   "--runtime", str(args.runtime), "--max-rounds", str(args.max_rounds),
                   "--interval", str(args.interval), "--steps", str(args.steps),
                   "--seconds", str(args.seconds), "--network", str(args.network)]
        command.extend(["--max-runtime-mb", str(args.max_runtime_mb)])
        if args.no_local_conversation:
            command.append("--no-local-conversation")
        process = subprocess.Popen(command, cwd=Path(__file__).parent, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True)
        print(json.dumps({"status": "started", "pid": process.pid,
                          "progress": str(args.runtime / "status.json")}, ensure_ascii=False))
        return
    result = work(args.seed, args.runtime, args.max_rounds, args.interval,
                  args.steps, args.seconds, args.network, not args.no_local_conversation,
                  args.max_runtime_mb)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
