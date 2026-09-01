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
import shutil
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
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
from visual_memory_v36 import (acquire_one as acquire_visual, empty_visual_memory,
                               enqueue as enqueue_visual, ground_depiction_labels)
from experience_revision_v37 import ExperienceRevisionEngine
from parser_self_revision_v38 import revise_parser
from parser_audit_memory_v39 import (empty_audit_memory, ingest_report as audit_parser_report,
                                     mark_curriculum_admission, rebuild_audit)
from micro_world_v41 import empty_world_memory, learn_steps as learn_micro_world
from tool_world_v42 import empty_tool_memory, learn_episodes as learn_tool_world


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
MAX_DEVELOPMENTAL_SHELF_DEPTH = 8
MAX_SHELF_PAGES_PER_DISCOVERY = 2000
MAX_MASTERY_HISTORY = 500
GIB = 1024 ** 3
DEFAULT_COMPACTION_BYTES = 20 * GIB
DEFAULT_MIN_FREE_BYTES = 50 * GIB
DEFAULT_RESUME_FREE_BYTES = 60 * GIB
DEFAULT_ABNORMAL_GROWTH_BYTES_PER_HOUR = 3 * GIB
DEVELOPMENTAL_SHELVES = (
    ("Category:Fables", 4.0),
    ("Category:Fairy tales", 3.0),
    ("Category:Children's literature", 2.0),
    ("Category:Folklore", 1.0),
)

PHASE_JA = {
    "starting": "起動中", "learning": "学習中", "between_rounds": "次の処理を準備中",
    "curriculum_transition": "次の教材へ移動中", "transient_error_wait": "一時エラーから再試行待ち",
    "supervisor_retry_wait": "監督機構による再試行待ち", "resource_paused": "外部取得の再開待ち",
    "storage_check": "容量確認中", "curriculum_exhausted": "教材候補を再探索中",
    "worker_error_wait": "内部エラーから復旧待ち", "stopped_by_user": "ユーザー操作で停止",
    "round_budget_exhausted": "指定回数を完了", "error": "エラー停止",
}
DIMENSION_JA = {"characters": "文字", "words": "単語", "phrases": "フレーズ",
                "conversation": "会話", "prediction": "予測", "causality": "因果",
                "concepts": "概念", "associations": "連想"}


def human_bytes(value: int | float | None) -> str:
    if value is None:
        return "不明"
    number = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(number) < 1024 or unit == "TB":
            return f"{number:.1f}{unit}" if unit != "B" else f"{int(number)}B"
        number /= 1024
    return f"{number:.1f}TB"


def percent(correct: int | float, total: int | float) -> str:
    return "評価前" if not total else f"{100 * correct / total:.1f}%"


def render_human_status(status: dict, now_epoch: float | None = None,
                        process_alive: bool | None = None) -> str:
    now_epoch = time.time() if now_epoch is None else now_epoch
    pid = status.get("pid")
    if process_alive is None:
        try:
            os.kill(int(pid), 0)
            process_alive = True
        except (OSError, TypeError, ValueError):
            process_alive = False
    heartbeat = status.get("heartbeat")
    heartbeat_epoch = None
    heartbeat_ja = "不明"
    if heartbeat:
        try:
            stamp = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
            heartbeat_epoch = stamp.timestamp()
            heartbeat_ja = stamp.astimezone(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S JST")
        except ValueError:
            pass
    age = None if heartbeat_epoch is None else max(0, int(now_epoch - heartbeat_epoch))
    stale = age is None or age > 120
    phase = status.get("phase", "unknown")
    healthy = process_alive and not stale and phase not in {"error", "stopped_by_user"}
    health = "正常に稼働" if healthy else "確認が必要"
    global_memory = status.get("global_memory", {})
    mastery = status.get("mastery", {})
    association = status.get("association", {})
    association_eval = association.get("evaluation", {})
    causal = status.get("causal_evaluation", {})
    causal_eval = causal.get("evaluation", {})
    representation = status.get("representation", {}).get("selected_evaluation", {})
    errors = status.get("error_memory", {})
    visual = status.get("visual_memory", {})
    revision = status.get("experience_revision", {})
    revision_eval = revision.get("evaluation", {})
    parser_revision = status.get("parser_revision", {})
    parser_eval = parser_revision.get("selected_evaluation", {}) or {}
    parser_audit = status.get("parser_audit", {})
    autonomy = status.get("autonomy", {})
    micro_world = status.get("micro_world", {})
    tool_world = status.get("tool_world", {})
    scaffold = status.get("epistemic_scaffold", {})
    storage = status.get("storage", {})
    quality = status.get("developmental_quality")
    lines = ["Noise 学習状況", "=" * 34,
             f"総合判定       : {health}",
             f"動作状態       : {PHASE_JA.get(phase, phase)}",
             f"プロセス       : {'生存' if process_alive else '停止'}（PID {pid or '不明'}）",
             f"最終更新       : {heartbeat_ja}" + (f"（{age}秒前）" if age is not None else ""),
             f"現在の教材     : {status.get('seed') or '不明'}",
             f"起動後ラウンド : {status.get('rounds', 0)}",
             f"外部LLM利用    : {status.get('codex_or_remote_llm_calls', 0)}回",
             f"自律運転       : {autonomy.get('mode', '評価中')}（人の操作 {'必要' if autonomy.get('human_intervention_required') else '不要'}）",
             "", "言語と経験", "-" * 34,
             f"採用教材       : {global_memory.get('curricula', 0):,}",
             f"単語           : {global_memory.get('word_forms', 0):,}（根拠あり {global_memory.get('grounded_word_forms', 0):,}）",
             f"フレーズ       : {global_memory.get('phrases', 0):,}（根拠あり {global_memory.get('grounded_phrases', 0):,}）",
             f"品質確認イベント: {global_memory.get('quality_events', 0):,}",
             f"人間科学観測   : {scaffold.get('observation_frames', 0):,}件（解釈 {scaffold.get('interpretations_committed', 0)}、仮説 {scaffold.get('hypotheses_committed', 0)}）",
             "", "能動実験世界", "-" * 34,
             f"第一段階       : {micro_world.get('status', '準備中')}",
             f"自分で行った実験: {micro_world.get('interventions', 0):,}回",
             f"予測失敗       : {micro_world.get('prediction_errors', 0):,}回",
             f"規則修正       : {micro_world.get('corrective_revisions', 0):,}回",
             f"残った仮説     : {micro_world.get('surviving_hypotheses', 0):,}個",
             f"未見世界評価   : {micro_world.get('holdout', {}).get('correct', 0)}/{micro_world.get('holdout', {}).get('total', 0)}（{percent(micro_world.get('holdout', {}).get('correct', 0), micro_world.get('holdout', {}).get('total', 0))}）",
             f"第二段階       : {tool_world.get('status', '第一段階の合格待ち')}",
             f"道具世界試行   : {tool_world.get('episodes', 0):,}回",
             f"成功した計画   : {tool_world.get('successful_training_plans', 0):,}件",
             f"行動失敗の記憶 : {tool_world.get('remembered_action_failures', 0):,}件",
             f"未見道具課題   : {tool_world.get('unseen_tasks', {}).get('successes', 0)}/{tool_world.get('unseen_tasks', {}).get('total', 0)}（{percent(tool_world.get('unseen_tasks', {}).get('successes', 0), tool_world.get('unseen_tasks', {}).get('total', 0))}）",
             "", "現在の能力評価", "-" * 34]
    ac, at = association_eval.get("correct", 0), association_eval.get("total", 0)
    ab = association_eval.get("baseline_correct", 0)
    association_judgement = "基準を上回った" if ac > ab else ("基準と同じ" if ac == ab else "基準より下")
    cc, ct = causal_eval.get("correct", 0), causal_eval.get("total", 0)
    cb = causal_eval.get("baseline_correct", 0)
    causal_judgement = "基準を上回った" if cc > cb else ("基準と同じ" if cc == cb else "基準より下")
    lines.extend([
        f"連想予測       : {ac}/{at}（{percent(ac, at)}）、単純基準 {ab}/{at} → {association_judgement}",
        f"連想の修正     : 強化 {association.get('reinforced', 0)}、弱化 {association.get('weakened', 0)}",
        f"因果予測       : {cc}/{ct}（{percent(cc, ct)}）、単純基準 {cb}/{ct} → {causal_judgement}",
        f"因果候補       : {causal.get('supported_hypotheses', 0)}件（まだ証明ではない）",
        f"抽象表現       : 正解 {representation.get('correct', 0)}/{representation.get('total', 0)}、適用範囲 {100 * representation.get('coverage', 0):.1f}%",
        f"構造規則       : {revision.get('rules_formed', 0):,}件（再利用可能 {revision.get('reusable_rules', 0):,}、弱化 {revision.get('weakened_rules', 0):,}）",
        f"構造予測       : {revision_eval.get('correct', 0)}/{revision_eval.get('total', 0)}（{percent(revision_eval.get('correct', 0), revision_eval.get('total', 0))}）、適用範囲 {100 * revision_eval.get('coverage', 0):.1f}%",
        f"失敗原因分析   : {revision.get('prediction_errors', 0):,}件",
        f"解析方式       : {parser_revision.get('selected_policy') or 'baseline'}（{parser_revision.get('selection_status') or '評価前'}）",
        f"解析方式の評価 : 正解 {parser_eval.get('correct', 0)}/{parser_eval.get('total', 0)}、解析範囲 {100 * parser_eval.get('parse_coverage', 0):.1f}%",
        f"解析監査       : {parser_audit.get('audited_sentences', 0):,}文（隔離した不採用 {parser_audit.get('quarantined_rejections', 0):,}）",
        f"最優先の弱点   : {DIMENSION_JA.get(mastery.get('weakest_dimension'), mastery.get('weakest_dimension') or '未判定')}",
        "", "間違いの記憶", "-" * 34,
        f"認識した誤り   : {errors.get('recognized_errors', 0):,}件",
        f"再発した誤り   : {errors.get('repeated_errors', 0):,}件",
        f"訂正に反映済み : {errors.get('corrective_changes', 0):,}件",
        f"現在有効な訂正 : {errors.get('currently_corrected', errors.get('corrective_changes', 0)):,}件",
        f"反例として保持 : {errors.get('unresolved_errors', 0):,}件",
        "", "画像経験", "-" * 34,
        f"画像表現を観測 : {visual.get('depictions_seen', 0):,}枚",
        f"視覚待ち教材   : {visual.get('pending_visual_curricula', 0):,}",
        f"実物を観測     : {visual.get('physical_objects_seen', 0):,}件",
        f"接地済み概念   : {visual.get('grounded_visual_concepts', 0):,}件",
        "", "ストレージ", "-" * 34,
        f"永続データ     : {human_bytes(storage.get('runtime_bytes'))}",
        f"再取得可能キャッシュ: {human_bytes(storage.get('reconstructible_cache_bytes'))}",
        f"管理対象合計   : {human_bytes(storage.get('managed_bytes', storage.get('after_bytes')))}",
        f"ディスク空き   : {human_bytes(storage.get('disk_free_bytes'))}",
        f"10GB警告       : {'発生中' if storage.get('warning') else 'なし'}",
        f"外部取得停止   : {'停止中: ' + ', '.join(storage.get('pause_reasons', [])) if storage.get('external_acquisition_paused') else 'なし'}",
    ])
    if quality:
        admitted = status.get("global_memory_admission", {}).get("admitted")
        lines.extend(["", "直近の教材審査", "-" * 34,
                      f"判定           : {'長期記憶へ採用' if admitted else '不採用（記憶を汚さず次へ）'}",
                      f"適合スコア     : {quality.get('score', 0):.3f}"])
    notes = []
    if not process_alive:
        notes.append("ワーカープロセスが停止しています。")
    if stale:
        notes.append("最終更新が2分以上前です。停止または処理詰まりを確認してください。")
    if status.get("error"):
        notes.append(f"エラー: {status['error']}")
    if ac <= ab:
        notes.append("連想はまだ単純基準を上回っていません。経験を追加しながら修正中です。")
    if cc <= cb:
        notes.append("因果予測はまだ単純基準を上回っていません。")
    lines.extend(["", "要点", "-" * 34, *[f"・{note}" for note in notes]])
    return "\n".join(lines)


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
    """Apply the staged storage policy; max_bytes is the compaction threshold."""
    previous = read_json(runtime / "storage-status.json")
    checked_epoch = time.time()
    runtime_before = runtime_bytes(runtime)
    cache_dir = WEB_CACHE.cache_dir if runtime.resolve() == DEFAULT_RUNTIME.resolve() else None
    cache_before = runtime_bytes(cache_dir) if cache_dir and cache_dir.exists() else 0
    before = runtime_before + cache_before
    warning_bytes = max_bytes // 2
    disk_free = shutil.disk_usage(runtime).free
    previous_epoch = previous.get("checked_epoch")
    # Old v26 records counted only .local; do not mistake newly included cache bytes for growth.
    previous_managed = previous.get("managed_bytes")
    elapsed_hours = ((checked_epoch - previous_epoch) / 3600
                     if previous_epoch and checked_epoch > previous_epoch else None)
    growth = (before - previous_managed if previous_managed is not None else None)
    growth_per_hour = (growth / elapsed_hours if elapsed_hours and growth is not None else None)
    abnormal_growth = bool(growth_per_hour is not None
                           and growth_per_hour >= DEFAULT_ABNORMAL_GROWTH_BYTES_PER_HOUR)
    previous_paused = bool(previous.get("external_acquisition_paused"))
    previous_reasons = set(previous.get("pause_reasons", []))
    low_free = disk_free < DEFAULT_MIN_FREE_BYTES
    low_free_hold = (previous_paused and "low_disk_free" in previous_reasons
                     and disk_free < DEFAULT_RESUME_FREE_BYTES)
    pause_reasons = []
    if low_free or low_free_hold:
        pause_reasons.append("low_disk_free")
    if abnormal_growth:
        pause_reasons.append("abnormal_growth")
    compacted = None
    last_compaction_epoch = previous.get("last_compaction_epoch")
    compaction_due = (before >= max_bytes and
                      (last_compaction_epoch is None
                       or checked_epoch - last_compaction_epoch >= 24 * 3600))
    if compaction_due:
        compacted = compact_runtime(runtime, True)
        last_compaction_epoch = checked_epoch
    runtime_after = runtime_bytes(runtime)
    cache_after = runtime_bytes(cache_dir) if cache_dir and cache_dir.exists() else 0
    after = runtime_after + cache_after
    record = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(checked_epoch)),
              "checked_epoch": checked_epoch,
              "runtime_bytes": runtime_after, "reconstructible_cache_bytes": cache_after,
              "managed_bytes": after, "before_bytes": before, "after_bytes": after,
              "warning_bytes": warning_bytes, "compaction_bytes": max_bytes,
              "limit_bytes": max_bytes, "warning": after >= warning_bytes,
              "compaction_recommended": after >= max_bytes,
              "compaction_due": compaction_due,
              "compacted": compacted is not None,
              "last_compaction_epoch": last_compaction_epoch,
              "bytes_reclaimed": 0 if not compacted else compacted["bytes_reclaimed"],
              "disk_free_bytes": disk_free,
              "minimum_free_bytes": DEFAULT_MIN_FREE_BYTES,
              "resume_free_bytes": DEFAULT_RESUME_FREE_BYTES,
              "growth_bytes_since_check": growth,
              "growth_bytes_per_hour": None if growth_per_hour is None else round(growth_per_hour),
              "abnormal_growth_bytes_per_hour": DEFAULT_ABNORMAL_GROWTH_BYTES_PER_HOUR,
              "abnormal_growth": abnormal_growth,
              "external_acquisition_paused": bool(pause_reasons),
              "pause_reasons": pause_reasons,
              "protected_memories": ["global-language-memory", "error-memory",
                                       "epistemic-observations", "visual-memory",
                                       "micro-world-memory", "tool-world-memory"]}
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


def update_autonomy_state(curriculum: dict, report: dict) -> dict:
    """Detect a measured plateau and change observation strategy without human prompting."""
    revision = report.get("experience_revision", {})
    global_memory = report.get("global_memory", {})
    snapshot = {"curricula": global_memory.get("curricula", 0),
                "structural_correct": revision.get("evaluation", {}).get("correct", 0),
                "structural_total": revision.get("evaluation", {}).get("total", 0),
                "structural_coverage": revision.get("evaluation", {}).get("coverage", 0.0),
                "reusable_rules": revision.get("reusable_rules", 0),
                "failure_patterns": len(revision.get("failure_patterns", []))}
    history = curriculum.setdefault("capability_history", [])
    if not history or history[-1].get("curricula") != snapshot["curricula"]:
        history.append(snapshot)
        curriculum["capability_history"] = history[-200:]
    window = curriculum["capability_history"][-30:]
    plateau = (len(window) >= 10
               and window[-1]["curricula"] - window[0]["curricula"] >= 20
               and window[-1]["structural_total"] > window[0]["structural_total"]
               and window[-1]["structural_correct"] <= window[0]["structural_correct"])
    state = {"mode": "counterexample_hunt" if plateau else "normal_curriculum",
             "plateau_detected": plateau, "observations_compared": len(window),
             "reason": ("structural tests grew without another correct prediction" if plateau else
                        "no sustained measured plateau in the current window"),
             "human_intervention_required": False}
    curriculum["autonomy_state"] = state
    return state


def update_curriculum_strategy(curriculum: dict, seed: str, admitted: bool) -> dict:
    """Learn which candidate-generation routes actually yield developmental material."""
    transition = next((item for item in reversed(curriculum.get("transitions", []))
                       if item.get("to") == seed), None)
    strategy = (transition or {}).get("reason", "initial_or_external_seed")
    ledger = curriculum.setdefault("strategy_performance", {})
    item = ledger.setdefault(strategy, {"attempts": 0, "admitted": 0, "rejected": 0})
    outcomes = item.setdefault("seed_outcomes", {})
    previous = outcomes.get(seed)
    if previous is None:
        item["attempts"] += 1
        item["admitted" if admitted else "rejected"] += 1
    elif previous != admitted:
        item["admitted" if previous else "rejected"] -= 1
        item["admitted" if admitted else "rejected"] += 1
    outcomes[seed] = admitted
    item["admission_rate"] = round(item["admitted"] / item["attempts"], 4)
    item["status"] = ("deprioritized" if item["attempts"] >= 10
                      and item["admission_rate"] < 0.2 else "active")
    return {"strategy": strategy, **item}


def curriculum_strategy_allowed(curriculum: dict, candidate: dict) -> bool:
    performance = curriculum.get("strategy_performance", {}).get(candidate.get("reason"), {})
    return performance.get("status") != "deprioritized"


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


def parser_counterexample_candidate(audit_memory: dict, visited: set[str]) -> dict | None:
    """Turn an actual quarantined failure into a request for a nearby observation."""
    records = [item for item in audit_memory.get("records", {}).values()
               if item.get("quarantined") and item.get("sentence")
               and item.get("curriculum_admitted") is True]
    reason_counts = audit_memory.get("summary", {}).get("rejection_reasons", {})
    ranked_reasons = [name for name, _ in sorted(reason_counts.items(),
                                                  key=lambda item: (-item[1], item[0]))]
    for reason in ranked_reasons:
        for item in reversed(records):
            if not str(item.get("reason", "")).startswith(reason):
                continue
            words = [word.lower() for word in WORD.findall(item["sentence"])
                     if word.lower() not in TITLE_STOP | CURRICULUM_METADATA and len(word) >= 3]
            seed = " ".join(list(dict.fromkeys(words))[:4])
            if seed and seed not in visited and valid_curriculum_seed(seed):
                return {"seed": seed, "score": 5.5,
                        "reason": f"seek a new observation resembling parser failure: {reason}",
                        "parent_url": item.get("source_url"),
                        "parser_failure_reason": reason,
                        "audit_id": item.get("audit_id")}
    return None


def structural_counterexample_candidate(experience_report: dict, visited: set[str]) -> dict | None:
    """Request another observation for the most frequent unresolved structural failure."""
    patterns = experience_report.get("summary", {}).get("failure_patterns", [])
    for item in patterns:
        terms = [term.lower() for term in item.get("query_terms", [])
                 if term and term.lower() not in TITLE_STOP | CURRICULUM_METADATA]
        seed = " ".join(list(dict.fromkeys(terms))[:4])
        if seed and seed not in visited and valid_curriculum_seed(seed):
            return {"seed": seed, "score": 6.0,
                    "reason": "seek an independent counterexample for a repeated structural failure",
                    "parent_url": None, "failure_pattern": item.get("pattern"),
                    "failure_count": item.get("count", 0)}
    return None


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
    shelf_pages_examined = 0
    while (queue and len(found) < MAX_FRONTIER
           and shelf_pages_examined < MAX_SHELF_PAGES_PER_DISCOVERY):
        shelf, shelf_score, depth, continuation = queue.pop(0)
        page_key = (shelf, continuation)
        if page_key in seen_pages or depth > MAX_DEVELOPMENTAL_SHELF_DEPTH:
            continue
        seen_pages.add(page_key)
        seen_shelves.add(shelf)
        shelf_pages_examined += 1
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
    visual_memory = read_json(runtime / "visual-memory.json")
    experience_revision = read_json(runtime / "experience-revision.json")
    parser_revision = read_json(runtime / "parser-revision.json")
    parser_audit = read_json(runtime / "parser-audit-memory.json")
    micro_world = read_json(runtime / "micro-world-memory.json")
    tool_world = read_json(runtime / "tool-world-memory.json")
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
        "visual_memory": report.get("visual_memory") or visual_memory.get("summary", {}),
        "experience_revision": report.get("experience_revision") or
                               experience_revision.get("summary", {}),
        "parser_revision": report.get("parser_revision") or {
            key: parser_revision.get(key) for key in
            ("selected_policy", "selection_status", "selected_evaluation",
             "failure_causes", "revisions")},
        "parser_audit": report.get("parser_audit") or parser_audit.get("summary", {}),
        "autonomy": report.get("autonomy") or read_json(
            runtime / "curriculum-state.json").get("autonomy_state", {}),
        "micro_world": report.get("micro_world") or micro_world.get("summary", {}),
        "tool_world": report.get("tool_world") or tool_world.get("summary", {}),
        "visual_observation": report.get("visual_observation"),
        "developmental_quality": report.get("developmental_quality"),
        "global_memory_admission": report.get("global_memory_admission"),
    }


def work(seed: str, runtime: Path, max_rounds: int, interval: float,
         steps: int, seconds: float, network: int, local_conversation: bool = True,
         max_runtime_mb: int = 20 * 1024) -> dict:
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
    storage_status = enforce_storage_budget(runtime, max_runtime_mb * 1024 * 1024)
    parser_policy = read_json(runtime / "parser-revision.json").get("selected_policy", "baseline")
    os.environ["AI_NOISE_PARSER_POLICY"] = parser_policy
    latest = status_record(seed, runtime, "starting", 0)
    write_json(status_path, latest)
    round_number = 0
    consecutive_transient_errors = 0
    while max_rounds <= 0 or round_number < max_rounds:
        round_number += 1
        if round_number > 1 and round_number % 100 == 0:
            write_json(status_path, status_record(seed, runtime, "storage_check", round_number - 1))
            storage_status = enforce_storage_budget(runtime, max_runtime_mb * 1024 * 1024)
        if stop_path.exists():
            latest = status_record(seed, runtime, "stopped_by_user", round_number - 1)
            write_json(status_path, latest)
            return latest
        write_json(status_path, status_record(seed, runtime, "learning", round_number - 1))
        effective_network = 0 if storage_status.get("external_acquisition_paused") else network
        micro_path = runtime / "micro-world-memory.json"
        micro_memory = read_json(micro_path) or empty_world_memory()
        micro_summary = learn_micro_world(micro_memory, 3)
        write_json(micro_path, micro_memory)
        tool_path = runtime / "tool-world-memory.json"
        tool_memory = read_json(tool_path) or empty_tool_memory()
        tool_summary = (learn_tool_world(tool_memory, 25)
                        if micro_summary.get("status") == "stage_1_mastered" else
                        {"stage": 2, "status": "waiting_for_stage_1"})
        write_json(tool_path, tool_memory)
        try:
            report = run_cycle(seed, runtime, steps, seconds, effective_network,
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
        report["micro_world"] = micro_summary
        report["tool_world"] = tool_summary
        reason = report.get("state", {}).get("stop_reason")
        audit_path = runtime / "parser-audit-memory.json"
        parser_audit_memory = read_json(audit_path) or rebuild_audit(runtime)
        audit_parser_report(parser_audit_memory, seed, report)
        write_json(audit_path, parser_audit_memory)
        report["parser_audit"] = parser_audit_memory.get("summary", {})
        memory_path = runtime / "global-language-memory.json"
        memory = read_json(memory_path) or empty_memory()
        developmentally_known_words = {form for form, item in memory.get("words", {}).items()
                                       if item.get("curricula", 0) >= 3}
        report["developmental_quality"] = assess_source_quality(
            report, developmentally_known_words)
        japanese_grounded = bool(JAPANESE.search(seed) and report.get("knowledge", {}).get(
            "lexicon", {}).get("grounded_meanings"))
        admitted = report["developmental_quality"].get("admit_to_global_memory") or japanese_grounded
        report["curriculum_strategy"] = update_curriculum_strategy(curriculum, seed, bool(admitted))
        report["global_memory_admission"] = {
            "admitted": bool(admitted),
            "reason": ("audited developmental passage" if report["developmental_quality"].get(
                "admit_to_global_memory") else ("grounded Japanese boundary path" if japanese_grounded
                else "outside current developmental level; raw report retained"))}
        mark_curriculum_admission(parser_audit_memory, seed, admitted)
        write_json(audit_path, parser_audit_memory)
        report["parser_audit"] = parser_audit_memory.get("summary", {})
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
        previous_parser = read_json(runtime / "parser-revision.json")
        if new_global_experience or not previous_parser:
            parser_report = revise_parser(
                scaffold.get("frames", {}), previous_parser,
                parser_audit_memory.get("summary", {}),
                read_json(runtime / "experience-revision.json").get("summary", {}))
            write_json(runtime / "parser-revision.json", parser_report)
            os.environ["AI_NOISE_PARSER_POLICY"] = parser_report["selected_policy"]
        else:
            parser_report = previous_parser
        report["parser_revision"] = {
            key: parser_report.get(key) for key in
            ("selected_policy", "selection_status", "selected_evaluation",
             "failure_causes", "revisions")}
        visual_path = runtime / "visual-memory.json"
        visual = read_json(visual_path) or empty_visual_memory()
        enqueue_visual(visual, list(memory.get("merged_seeds", [])))
        visual_result = ({"status": "storage_policy_paused", **visual.get("summary", {})}
                         if storage_status.get("external_acquisition_paused") else
                         acquire_visual(visual, runtime / "visual" / "images"))
        grounded_forms = {form for form, item in memory.get("words", {}).items()
                          if item.get("curricula", 0) >= 3}
        visual_result["language_grounding"] = ground_depiction_labels(visual, grounded_forms)
        write_json(visual_path, visual)
        report["visual_memory"] = visual.get("summary", {})
        report["visual_observation"] = visual_result
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
            experience_report = ExperienceRevisionEngine(
                memory.get("quality_event_transitions", {})).run()
            write_json(runtime / "experience-revision.json", experience_report)
        else:
            causal_report = read_json(runtime / "causal-memory.json")
            representation_report = read_json(runtime / "representation-memory.json")
            association_report = read_json(runtime / "association-memory.json")
            experience_report = read_json(runtime / "experience-revision.json")
            if not association_report:
                association_report = AssociationLearner(
                    memory.get("quality_event_transitions", {}),
                    memory.get("quality_event_counts", {})).run()
                write_json(runtime / "association-memory.json", association_report)
            if not experience_report:
                experience_report = ExperienceRevisionEngine(
                    memory.get("quality_event_transitions", {})).run()
                write_json(runtime / "experience-revision.json", experience_report)
        error_path = runtime / "error-memory.json"
        error_ledger = read_json(error_path) or empty_error_memory()
        update_error_memory(error_ledger, association_report, causal_report,
                            memory.get("totals", {}).get("curricula", 0), experience_report)
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
        report["experience_revision"] = experience_report.get("summary", {})
        report["autonomy"] = update_autonomy_state(curriculum, report)
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
            discovered = discover_curriculum(report, visited, effective_network)
            if report.get("autonomy", {}).get("mode") == "counterexample_hunt":
                targeted = structural_counterexample_candidate(experience_report, visited)
                if targeted:
                    discovered.insert(0, targeted)
            known = {item["seed"] for item in curriculum["frontier"]}
            curriculum["frontier"].extend(item for item in discovered if item["seed"] not in known)
            curriculum["frontier"] = [item for item in curriculum["frontier"]
                                      if item["seed"] not in visited
                                      and curriculum_strategy_allowed(curriculum, item)
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
                curriculum["frontier"].extend(
                    rediscover_from_history(runtime, visited, effective_network))
            if not curriculum["frontier"]:
                structural_candidate = structural_counterexample_candidate(
                    experience_report, visited)
                if structural_candidate:
                    curriculum["frontier"].append(structural_candidate)
            if not curriculum["frontier"]:
                parser_candidate = parser_counterexample_candidate(parser_audit_memory, visited)
                if parser_candidate:
                    curriculum["frontier"].append(parser_candidate)
            if not curriculum["frontier"]:
                curriculum["frontier"].extend(
                    discover_from_developmental_shelves(visited, effective_network))
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
              max_runtime_mb: int = 20 * 1024) -> dict:
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
        work_parser.add_argument("--max-runtime-mb", type=int, default=20 * 1024,
                                 help="redundant-state compaction threshold (default: 20480 MB)")
    run_parser = subparsers.add_parser("run", help="run in the foreground")
    add_work_arguments(run_parser)
    start_parser = subparsers.add_parser("start", help="start once and keep working in the background")
    add_work_arguments(start_parser)
    supervise_parser = subparsers.add_parser("supervise", help=argparse.SUPPRESS)
    add_work_arguments(supervise_parser)
    status_parser = subparsers.add_parser("status", help="show the last local heartbeat")
    status_parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    status_ja_parser = subparsers.add_parser("status-ja", help="show an easy Japanese status summary")
    status_ja_parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    stop_parser = subparsers.add_parser("stop", help="request a safe stop between cycles")
    stop_parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    args = parser.parse_args()

    if args.command == "status":
        print(json.dumps(read_json(args.runtime / "status.json"), ensure_ascii=False, indent=2))
        return
    if args.command == "status-ja":
        print(render_human_status(read_json(args.runtime / "status.json")))
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
