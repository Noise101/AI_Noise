#!/usr/bin/env python3
"""Measure whether a source fits the learner's current developmental level."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def assess_source_quality(report: dict, known_words: set[str] | None = None) -> dict:
    sources = report.get("knowledge", {}).get("bootstrap", {}).get("sources", [])
    audits = [item for source in sources for item in source.get("event_extraction_audit", [])]
    source_urls = sorted({source.get("url") for source in sources if source.get("url")})
    if not audits:
        return {"status": "not_yet_audited", "admit_to_global_memory": False,
                "accepted": 0, "total": 0, "score": 0.0, "source_urls": source_urls,
                "reasons": ["no v29 sentence audit"]}
    accepted_items = [item for item in audits if item.get("accepted") and item.get("event")]
    accepted = len(accepted_items)
    narrative_ratio = accepted / len(audits)
    sentences = [item.get("sentence", "") for item in accepted_items]
    sentence_words = [WORD.findall(sentence.lower()) for sentence in sentences]
    short_ratio = sum(len(words) <= 18 for words in sentence_words) / max(1, accepted)
    all_words = [word for words in sentence_words for word in words]
    known_words = known_words or set()
    known_ratio = (sum(word in known_words for word in all_words) / len(all_words)
                   if all_words and known_words else 0.5)
    subjects = [item["event"].split("|", 1)[0] for item in accepted_items]
    recurrence = 0.0 if not subjects else 1 - len(set(subjects)) / len(subjects)
    dialogue_ratio = sum(('"' in sentence or "“" in sentence or "?" in sentence)
                         for sentence in sentences) / max(1, accepted)
    score = (0.35 * narrative_ratio + 0.25 * short_ratio + 0.15 * known_ratio
             + 0.15 * recurrence + 0.10 * dialogue_ratio)
    reasons = []
    if accepted < 2:
        reasons.append("fewer than two accepted events")
    if narrative_ratio < 0.3:
        reasons.append("most audited sentences are not narrative events")
    if short_ratio < 0.5:
        reasons.append("sentences exceed the current child-level length")
    if score < 0.58:
        reasons.append("combined developmental score below 0.58")
    admitted = accepted >= 2 and narrative_ratio >= 0.3 and short_ratio >= 0.5 and score >= 0.58
    return {"status": "developmental_passage" if admitted else "outside_current_level",
            "admit_to_global_memory": admitted, "score": round(score, 3),
            "accepted": accepted, "total": len(audits),
            "metrics": {"narrative_ratio": round(narrative_ratio, 3),
                        "short_sentence_ratio": round(short_ratio, 3),
                        "known_word_ratio": round(known_ratio, 3),
                        "subject_recurrence": round(recurrence, 3),
                        "dialogue_ratio": round(dialogue_ratio, 3)},
            "source_urls": source_urls, "reasons": reasons}


def rebuild_developmental_memory(runtime: Path) -> dict:
    """Rebuild the canonical memory from admitted raw reports; retain the old memory as archive."""
    from global_memory_v27 import empty_memory, merge_report

    memory_path = runtime / "global-language-memory.json"
    old = json.loads(memory_path.read_text(encoding="utf-8")) if memory_path.exists() else empty_memory()
    known = {form for form, item in old.get("words", {}).items() if item.get("curricula", 0) >= 3}
    archive = runtime / "archive" / "global-language-memory-pre-v32.json"
    if memory_path.exists() and not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(memory_path, archive)
    clean = empty_memory()
    paths = [runtime / "latest-report.json", *sorted((runtime / "seeds").glob("*/latest-report.json"))]
    admitted = rejected = japanese = 0
    admitted_seeds, rejected_seeds = set(), set()
    trusted_urls, blocked_urls = set(), set()
    for path in paths:
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        seed = report.get("state", {}).get("seed")
        if not seed:
            continue
        quality = assess_source_quality(report, known)
        japanese_grounded = bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", seed)
                                 and report.get("knowledge", {}).get("lexicon", {}).get(
                                     "grounded_meanings"))
        if quality["admit_to_global_memory"] or japanese_grounded:
            report["global_memory_admission"] = {"admitted": True, "migration": "v32"}
            merge_report(clean, seed, report)
            admitted += 1
            admitted_seeds.add(seed)
            trusted_urls.update(quality.get("source_urls", []))
            japanese += int(japanese_grounded)
        else:
            rejected += 1
            rejected_seeds.add(seed)
            blocked_urls.update(quality.get("source_urls", []))
    temporary = memory_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(clean, ensure_ascii=False, separators=(",", ":")) + "\n",
                         encoding="utf-8")
    temporary.replace(memory_path)
    curriculum_path = runtime / "curriculum-state.json"
    curriculum_archive = runtime / "archive" / "curriculum-state-pre-v32.json"
    if curriculum_path.exists():
        if not curriculum_archive.exists():
            shutil.copy2(curriculum_path, curriculum_archive)
        curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
        curriculum["completed_seeds"] = sorted(admitted_seeds)
        curriculum["deferred_seeds"] = sorted(rejected_seeds - admitted_seeds)
        blocked_urls -= trusted_urls
        curriculum["trusted_parent_urls"] = sorted(trusted_urls)
        curriculum["blocked_parent_urls"] = sorted(blocked_urls)
        visited = admitted_seeds | rejected_seeds
        curriculum["frontier"] = [item for item in curriculum.get("frontier", [])
                                 if item.get("seed") not in visited
                                 and item.get("parent_url") not in blocked_urls][:300]
        curriculum_tmp = curriculum_path.with_suffix(".json.tmp")
        curriculum_tmp.write_text(json.dumps(curriculum, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
        curriculum_tmp.replace(curriculum_path)
    return {"admitted_reports": admitted, "rejected_reports": rejected,
            "japanese_reports": japanese, "archive": str(archive), "totals": clean["totals"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=Path(__file__).resolve().parent.parent / ".local")
    parser.add_argument("--rebuild-memory", action="store_true")
    args = parser.parse_args()
    if args.rebuild_memory:
        print(json.dumps(rebuild_developmental_memory(args.runtime), ensure_ascii=False))


if __name__ == "__main__":
    main()
