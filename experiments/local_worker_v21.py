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
import traceback
import urllib.parse
from pathlib import Path

from web_cache import WEB_CACHE, NetworkBudgetExceeded
from curiosity_drive_v23 import curiosity_pressure
from mastery_drive_v24 import assess_language_mastery
from local_conversation_v25 import practice_once
from compact_runtime_v26 import compact_runtime
from global_memory_v27 import empty_memory, mastery_report, merge_report
from causal_experiment_v28 import CausalExperimentEngine
from causal_lab_v30 import run_lab
from representation_learning_v31 import evaluate_representations, transform_transitions
from developmental_curriculum_v32 import assess_source_quality
from association_learning_v33 import AssociationLearner
from epistemic_scaffold_v34 import observe_report, rebuild_scaffold, summarize as summarize_scaffold
from error_memory_v35 import empty_error_memory, update_error_memory


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME = ROOT / ".local"
WORD = re.compile(r"[A-Za-z]+")
JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
TITLE_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "three",
              "hundred", "aesop", "s", "fables"}
CURRICULUM_METADATA = {"index", "preface", "introduction", "appendix", "volume", "chapter",
                       "translator", "bibliography", "edition", "notes", "contents", "carving",
                       "history", "dictionary", "encyclopedia", "book", "part", "section",
                       "act", "scene"}
MAX_FRONTIER = 300
MAX_MASTERY_HISTORY = 500
DEVELOPMENTAL_SHELVES = (
    ("Category:Fables", 4.0),
    ("Category:Fairy tales", 3.0),
    ("Category:Children's literature", 2.0),
    ("Category:Folklore", 1.0),
)


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
    if any(marker in text for marker in ("http error 403", "http error 404", "http error 410")):
        return False
    markers = ("timeout", "timed out", "temporarily unavailable", "connection reset",
               "incompleteread", "incomplete read",
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


def conversation_practice_summary(runtime: Path) -> dict:
    turns = read_json(runtime / "dialogue-ledger.json").get("turns", [])
    evaluated = [turn for turn in turns if turn.get("practice_metrics")]
    successful = sum(turn["practice_metrics"].get("formed_followup")
                     and turn["practice_metrics"].get("relevant_token_overlap", 0) > 0
                     for turn in evaluated)
    return {"evaluated_turns": len(evaluated), "successful_followups": successful}


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
    if not 2 <= len(unique) <= 5 or set(unique) & CURRICULUM_METADATA:
        return None
    return " ".join(unique[:3])


def valid_curriculum_seed(seed: str, linked_title: str | None = None) -> bool:
    if JAPANESE.search(seed):
        parts = seed.split()
        return 1 <= len(parts) <= 3 and all(1 <= len(part) <= 12 for part in parts)
    return _seed_from_title(linked_title or seed) is not None


def compact_learning_history(curriculum: dict) -> None:
    history = curriculum.setdefault("mastery_history", [])
    if len(history) <= MAX_MASTERY_HISTORY:
        return
    removed = history[:-MAX_MASTERY_HISTORY]
    summary = curriculum.setdefault("mastery_history_summary", {
        "records": 0, "score_sum": 0.0, "weakest_dimensions": {}})
    for item in removed:
        summary["records"] += 1
        summary["score_sum"] += item.get("overall_score", 0.0)
        name = item.get("weakest_dimension", "unknown")
        summary["weakest_dimensions"][name] = summary["weakest_dimensions"].get(name, 0) + 1
    summary["mean_score"] = round(summary["score_sum"] / max(1, summary["records"]), 4)
    curriculum["mastery_history"] = history[-MAX_MASTERY_HISTORY:]


def developmental_source_quality(report: dict) -> dict:
    return assess_source_quality(report)


def discover_curriculum(report: dict, visited: set[str], network: int) -> list[dict]:
    """Generate next seeds from observed evidence links and learned concepts."""
    candidates: dict[str, dict] = {}
    sources = report.get("knowledge", {}).get("bootstrap", {}).get("sources", [])
    source_quality = developmental_source_quality(report)
    WEB_CACHE.set_network_budget(network)
    for source in sources if source_quality["status"] == "developmental_passage" else []:
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
    beliefs = (report.get("knowledge", {}).get("concepts", {}).get("beliefs", [])
               if source_quality["status"] == "developmental_passage" else [])
    for belief in beliefs:
        citations = set(belief.get("citations") or [])
        if (belief.get("status") != "corroborated" or belief.get("accepted_polarity") is not True
                or len(citations) < 2):
            continue
        seed = f"{belief.get('subject', '')} {belief.get('object', '')}".strip().lower()
        if seed in visited or len(seed.split()) < 2 or not valid_curriculum_seed(seed):
            continue
        candidates.setdefault(seed, {"seed": seed, "score": 0.5,
                                     "reason": "unvisited concept pair from evidence ledger",
                                     "parent_url": sorted(citations)[0],
                                     "evidence_status": "corroborated",
                                     "independent_sources": len(citations)})
    chunks = [item for item in report.get("knowledge", {}).get("lexicon", {}).get(
        "phrase_candidates", []) if item.get("kind") == "unsegmented_chunk_candidate"
        and JAPANESE.search(item.get("phrase", ""))
        and (source_quality["status"] == "developmental_passage"
             or JAPANESE.search(report.get("state", {}).get("seed", "")))]
    if chunks:
        forms = list(dict.fromkeys(item["phrase"] for item in chunks[:2]))
        seed = " ".join(forms)
        if seed not in visited and valid_curriculum_seed(seed):
            candidates.setdefault(seed, {"seed": seed, "score": 1.5,
                "reason": "repeated unsegmented Japanese chunks require boundary grounding",
                "parent_url": None})
    return sorted(candidates.values(), key=lambda item: (-item["score"], item["seed"]))[:MAX_FRONTIER]


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


def discover_from_developmental_shelves(visited: set[str], network: int) -> list[dict]:
    """Find unread child-level titles after the evidence-linked frontier is empty.

    Shelves provide titles only. Text still has to pass the developmental audit
    before it enters global memory or generates further curriculum.
    """
    WEB_CACHE.set_network_budget(network)
    found: dict[str, dict] = {}
    queue = [(shelf, score, 0, None) for shelf, score in DEVELOPMENTAL_SHELVES]
    seen_pages: set[tuple[str, str | None]] = set()
    seen_shelves: set[str] = set()
    while queue and len(found) < MAX_FRONTIER:
        shelf, shelf_score, depth, continuation = queue.pop(0)
        page_key = (shelf, continuation)
        if page_key in seen_pages or depth > 2:
            continue
        seen_pages.add(page_key)
        seen_shelves.add(shelf)
        params = urllib.parse.urlencode({"action": "query", "list": "categorymembers",
                                         "cmtitle": shelf, "cmtype": "page|subcat",
                                         "cmlimit": 100, "format": "json", "formatversion": 2,
                                         **({"cmcontinue": continuation} if continuation else {})})
        try:
            data = WEB_CACHE.get_json("https://en.wikisource.org/w/api.php?" + params,
                                      "AI_Noise/0.32 (read-only developmental shelf)")
        except Exception:
            continue
        next_page = data.get("continue", {}).get("cmcontinue")
        if next_page:
            queue.append((shelf, shelf_score, depth, next_page))
        for page in data.get("query", {}).get("categorymembers", []):
            title = page.get("title", "")
            if page.get("ns") == 14 or title.startswith("Category:"):
                if title not in seen_shelves:
                    queue.append((title, max(0.5, shelf_score - 0.25), depth + 1, None))
                continue
            if title.count("/") > 1:
                continue
            seed = _seed_from_title(title)
            if not seed or seed in visited or seed in found:
                continue
            found[seed] = {"seed": seed, "score": shelf_score,
                           "reason": "unread title selected from a developmental shelf",
                           "parent_url": "https://en.wikisource.org/wiki/" +
                                         urllib.parse.quote(shelf.replace(" ", "_"), safe=":'_"),
                           "linked_title": title}
    return sorted(found.values(), key=lambda item: (-item["score"], item["seed"]))[:MAX_FRONTIER]


def status_record(seed: str, runtime: Path, phase: str, rounds: int,
                  report: dict | None = None, error: str | None = None) -> dict:
    report = report or {}
    state = report.get("state", read_json(runtime / "controller-state.json"))
    mastery = report.get("mastery") or read_json(runtime / "mastery.json")
    causal_memory = read_json(runtime / "causal-memory.json")
    representation_memory = read_json(runtime / "representation-memory.json")
    association_memory = read_json(runtime / "association-memory.json")
    epistemic_scaffold = read_json(runtime / "epistemic-observations.json")
    error_memory = read_json(runtime / "error-memory.json")
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
        "causal_lab": report.get("causal_lab") or read_json(runtime / "causal-lab.json"),
        "representation": report.get("representation") or {
            key: representation_memory.get(key) for key in
            ("selected_scheme", "selection_status", "selected_evaluation", "revisions")},
        "association": report.get("association") or {
            "evaluation": association_memory.get("evaluation", {}),
            "reinforced": association_memory.get("reinforced", 0),
            "weakened": association_memory.get("weakened", 0),
            "warning": association_memory.get("warning"),
        },
        "epistemic_scaffold": report.get("epistemic_scaffold") or
                              epistemic_scaffold.get("summary", {}),
        "error_memory": report.get("error_memory") or error_memory.get("summary", {}),
        "developmental_quality": report.get("developmental_quality"),
        "global_memory_admission": report.get("global_memory_admission"),
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
        developmentally_known_words = {form for form, item in memory.get("words", {}).items()
                                       if item.get("curricula", 0) >= 3}
        report["developmental_quality"] = assess_source_quality(
            report, developmentally_known_words)
        japanese_grounded = bool(JAPANESE.search(seed) and report.get("knowledge", {}).get(
            "lexicon", {}).get("grounded_meanings"))
        admitted = report["developmental_quality"].get("admit_to_global_memory") or japanese_grounded
        report["global_memory_admission"] = {
            "admitted": bool(admitted),
            "reason": ("audited developmental passage" if report["developmental_quality"].get(
                "admit_to_global_memory") else ("grounded Japanese boundary path" if japanese_grounded
                else "outside current developmental level; raw report retained"))}
        new_global_experience = merge_report(memory, seed, report) if admitted else False
        write_json(memory_path, memory)
        scaffold_path = runtime / "epistemic-observations.json"
        scaffold = read_json(scaffold_path)
        if not scaffold:
            scaffold = rebuild_scaffold(runtime, set(memory.get("merged_seeds", [])))
        if admitted:
            observe_report(scaffold, seed, report)
        scaffold["summary"] = summarize_scaffold(scaffold)
        write_json(scaffold_path, scaffold)
        report["epistemic_scaffold"] = scaffold["summary"]
        if new_global_experience or not (runtime / "causal-memory.json").exists():
            previous_representation = read_json(runtime / "representation-memory.json")
            representation_report = evaluate_representations(
                memory.get("quality_event_transitions", {}))
            selected_evaluation = next((item for item in representation_report["evaluations"]
                if item["scheme"] == representation_report["selected_scheme"]), {})
            representation_report["selected_evaluation"] = selected_evaluation
            revisions = previous_representation.get("revisions", [])
            before_scheme = previous_representation.get("selected_scheme")
            if before_scheme and before_scheme != representation_report["selected_scheme"]:
                revisions.append({"before": before_scheme,
                    "after": representation_report["selected_scheme"],
                    "reason": "new holdout evidence changed predictive ranking",
                    "at_curricula": memory.get("totals", {}).get("curricula", 0)})
            representation_report["revisions"] = revisions[-100:]
            write_json(runtime / "representation-memory.json", representation_report)
            abstract_transitions = transform_transitions(
                memory.get("quality_event_transitions", {}), representation_report)
            # Legacy transitions predate extraction audits and remain quarantined from causal claims.
            causal_report = CausalExperimentEngine(abstract_transitions).run()
            write_json(runtime / "causal-memory.json", causal_report)
            association_report = AssociationLearner(
                memory.get("quality_event_transitions", {}),
                memory.get("quality_event_counts", {})).run()
            write_json(runtime / "association-memory.json", association_report)
        else:
            causal_report = read_json(runtime / "causal-memory.json")
            representation_report = read_json(runtime / "representation-memory.json")
            association_report = read_json(runtime / "association-memory.json")
            if not association_report:
                association_report = AssociationLearner(
                    memory.get("quality_event_transitions", {}),
                    memory.get("quality_event_counts", {})).run()
                write_json(runtime / "association-memory.json", association_report)
        error_path = runtime / "error-memory.json"
        error_ledger = read_json(error_path) or empty_error_memory()
        update_error_memory(error_ledger, association_report, causal_report,
                            memory.get("totals", {}).get("curricula", 0))
        write_json(error_path, error_ledger)
        mastery = assess_language_mastery(
            mastery_report(memory), causal_report, conversation_practice_summary(runtime),
            representation_report, association_report)
        report["mastery"] = mastery
        report["global_memory"] = memory.get("totals", {})
        report["causal_evaluation"] = {
            "supported_hypotheses": causal_report.get("supported_hypotheses", 0),
            "evaluation": causal_report.get("evaluation", {}),
            "limitations": causal_report.get("limitations", []),
        }
        report["representation"] = {
            "selected_scheme": representation_report.get("selected_scheme"),
            "selection_status": representation_report.get("selection_status"),
            "selected_evaluation": representation_report.get("selected_evaluation", {}),
            "revisions": representation_report.get("revisions", []),
        }
        report["association"] = {
            "evaluation": association_report.get("evaluation", {}),
            "reinforced": association_report.get("reinforced", 0),
            "weakened": association_report.get("weakened", 0),
            "warning": association_report.get("warning"),
        }
        report["error_memory"] = error_ledger.get("summary", {})
        report["causal_lab"] = run_lab(seed)
        write_json(runtime / "causal-lab.json", report["causal_lab"])
        write_json(runtime / "mastery.json", mastery)
        curriculum["mastery_history"].append({"seed": seed, "round": round_number,
                                               "overall_score": mastery["overall_score"],
                                               "weakest_dimension": mastery["weakest_dimension"],
                                               "next_mastery_goal": mastery["next_mastery_goal"]})
        compact_learning_history(curriculum)
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
            low_quality = not admitted
            bucket = ("deferred_seeds" if low_quality or
                      reason == "no_new_evidence_for_unresolved_gap" else "completed_seeds")
            if seed not in curriculum[bucket]:
                curriculum[bucket].append(seed)
            quality_urls = report["developmental_quality"].get("source_urls", [])
            url_bucket = "trusted_parent_urls" if admitted else "blocked_parent_urls"
            curriculum.setdefault(url_bucket, [])
            curriculum[url_bucket] = sorted(set(curriculum[url_bucket]) | set(quality_urls))
            visited = set(curriculum["completed_seeds"]) | set(curriculum["deferred_seeds"])
            discovered = discover_curriculum(report, visited, network)
            known = {item["seed"] for item in curriculum["frontier"]}
            curriculum["frontier"].extend(item for item in discovered if item["seed"] not in known)
            curriculum["frontier"] = [item for item in curriculum["frontier"]
                                      if item["seed"] not in visited
                                      and valid_curriculum_seed(
                                          item["seed"], item.get("linked_title"))
                                      and item.get("parent_url") not in set(
                                          curriculum.get("blocked_parent_urls", []))
                                      and (item.get("reason") !=
                                           "unvisited concept pair from evidence ledger"
                                           or (item.get("evidence_status") == "corroborated"
                                               and item.get("independent_sources", 0) >= 2))]
            curriculum["frontier"] = sorted(
                curriculum["frontier"], key=lambda item: (-item.get("score", 0), item["seed"]))[:MAX_FRONTIER]
            if not curriculum["frontier"]:
                curriculum["frontier"].extend(rediscover_from_history(runtime, visited, network))
            if not curriculum["frontier"]:
                curriculum["frontier"].extend(
                    discover_from_developmental_shelves(visited, network))
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


def supervise(seed: str, runtime: Path, max_rounds: int, interval: float,
              steps: int, seconds: float, network: int, local_conversation: bool = True,
              max_runtime_mb: int = 1024) -> dict:
    """Keep autonomous work alive unless STOP is explicitly requested."""
    stop_path = runtime / "STOP"
    retry_count = 0
    while True:
        try:
            result = work(seed, runtime, max_rounds, interval, steps, seconds, network,
                          local_conversation, max_runtime_mb)
        except Exception as error:
            curriculum = read_json(runtime / "curriculum-state.json")
            current_seed = curriculum.get("current_seed", seed)
            result = status_record(current_seed, runtime, "worker_error_wait", 0,
                                   error=f"{type(error).__name__}: {error}")
            result["traceback"] = traceback.format_exc()[-4000:]
        if result.get("phase") == "stopped_by_user" or stop_path.exists():
            return result
        if max_rounds > 0:
            return result
        retry_count += 1
        retry_seconds = min(300, max(10, 2 ** min(retry_count, 8)))
        waiting = dict(result)
        waiting["previous_phase"] = result.get("phase")
        waiting["phase"] = "supervisor_retry_wait"
        waiting["retry_in_seconds"] = retry_seconds
        waiting["supervisor_retries"] = retry_count
        write_json(runtime / "status.json", waiting)
        if not wait_for_retry(stop_path, retry_seconds):
            waiting["phase"] = "stopped_by_user"
            write_json(runtime / "status.json", waiting)
            return waiting


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
    supervise_parser = subparsers.add_parser("supervise", help=argparse.SUPPRESS)
    add_work_arguments(supervise_parser)
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
        command = [sys.executable, str(Path(__file__).resolve()), "supervise", args.seed,
                   "--runtime", str(args.runtime), "--max-rounds", str(args.max_rounds),
                   "--interval", str(args.interval), "--steps", str(args.steps),
                   "--seconds", str(args.seconds), "--network", str(args.network)]
        command.extend(["--max-runtime-mb", str(args.max_runtime_mb)])
        if args.no_local_conversation:
            command.append("--no-local-conversation")
        log_path = args.runtime / "worker.log"
        args.runtime.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=Path(__file__).parent, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
        log.close()
        print(json.dumps({"status": "started", "pid": process.pid,
                          "progress": str(args.runtime / "status.json")}, ensure_ascii=False))
        return
    runner = supervise if args.command == "supervise" else work
    result = runner(args.seed, args.runtime, args.max_rounds, args.interval,
                    args.steps, args.seconds, args.network, not args.no_local_conversation,
                    args.max_runtime_mb)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
