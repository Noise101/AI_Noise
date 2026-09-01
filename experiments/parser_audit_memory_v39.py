#!/usr/bin/env python3
"""Quarantine every parser decision, including rejected sentences."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


def empty_audit_memory() -> dict:
    return {"version": 39, "observed_seeds": [], "records": {}, "summary": {},
            "admission_to_language_memory": False,
            "warning": "rejected text is retained for parser evaluation, not accepted as knowledge"}


def ingest_report(memory: dict, seed: str, report: dict) -> bool:
    records = memory.setdefault("records", {})
    before = len(records)
    for source in report.get("knowledge", {}).get("bootstrap", {}).get("sources", []):
        url = source.get("url", "")
        for position, item in enumerate(source.get("event_extraction_audit", [])):
            sentence = item.get("sentence", "")
            identity = hashlib.sha256(
                f"{seed}\n{url}\n{position}\n{sentence}\n{item.get('event')}".encode()).hexdigest()[:20]
            records[identity] = {"audit_id": identity, "seed": seed, "source_url": url,
                                 "source_position": position, "sentence": sentence,
                                 "accepted": bool(item.get("accepted")),
                                 "event": item.get("event"), "reason": item.get("reason"),
                                 "quality": item.get("quality", 0.0),
                                 "quarantined": not bool(item.get("accepted"))}
    if seed not in set(memory.setdefault("observed_seeds", [])):
        memory["observed_seeds"].append(seed)
    memory["summary"] = summarize(memory)
    return len(records) > before


def summarize(memory: dict) -> dict:
    records = list(memory.get("records", {}).values())
    rejected = [item for item in records if not item.get("accepted")]
    reasons = Counter(item.get("reason", "unknown").split(":", 1)[0] for item in rejected)
    return {"audited_sentences": len(records),
            "accepted_sentences": len(records) - len(rejected),
            "quarantined_rejections": len(rejected),
            "rejection_reasons": dict(reasons.most_common()),
            "admitted_as_world_knowledge": 0}


def rebuild_audit(runtime: Path) -> dict:
    memory = empty_audit_memory()
    paths = [runtime / "latest-report.json", *sorted((runtime / "seeds").glob("*/latest-report.json"))]
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        seed = report.get("state", {}).get("seed")
        if seed:
            ingest_report(memory, seed, report)
    memory["summary"] = summarize(memory)
    return memory
