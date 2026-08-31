#!/usr/bin/env python3
"""Persistent cross-mechanism memory of recognized mistakes and corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def empty_error_memory() -> dict:
    return {"version": 35, "records": {}, "summary": {},
            "warning": "recognized mistakes are learning history, not evidence that a replacement is true"}


def identity(domain: str, context: str, asserted: str | None, observed: str | None) -> str:
    return hashlib.sha256(f"{domain}\n{context}\n{asserted}\n{observed}".encode()).hexdigest()[:20]


def _remember(ledger: dict, *, domain: str, context: str, asserted: str | None,
              observed: str | None, occurrences: int, evidence: dict,
              correction: dict) -> None:
    key = identity(domain, context, asserted, observed)
    records = ledger.setdefault("records", {})
    timestamp = now()
    record = records.get(key)
    status = "recognized_and_corrected" if correction.get("mechanism_changed") else "recognized_error"
    if record is None:
        records[key] = {"error_id": key, "domain": domain, "kind": "prediction_error",
                        "context": context, "asserted_or_predicted": asserted,
                        "observed": observed, "occurrences": occurrences,
                        "first_recorded_at": timestamp, "last_confirmed_at": timestamp,
                        "status": status, "evidence": evidence, "correction": correction,
                        "revision_history": [{"at": timestamp, "event": "error_recognized",
                                              "occurrences": occurrences, "status": status}]}
        return
    before = (record.get("occurrences", 0), record.get("status"), record.get("correction"))
    record["occurrences"] = max(record.get("occurrences", 0), occurrences)
    record["last_confirmed_at"] = timestamp
    record["evidence"] = evidence
    record["correction"] = correction
    record["status"] = status
    after = (record["occurrences"], status, correction)
    if after != before:
        record.setdefault("revision_history", []).append(
            {"at": timestamp, "event": "error_evidence_revised",
             "before_occurrences": before[0], "occurrences": record["occurrences"],
             "before_status": before[1], "status": status})


def summarize(ledger: dict) -> dict:
    records = list(ledger.get("records", {}).values())
    domains = Counter(item.get("domain", "unknown") for item in records)
    return {"recognized_errors": len(records),
            "recorded_occurrences": sum(item.get("occurrences", 0) for item in records),
            "repeated_errors": sum(item.get("occurrences", 0) > 1 for item in records),
            "corrective_changes": sum(item.get("status") == "recognized_and_corrected"
                                      for item in records),
            "unresolved_errors": sum(item.get("status") == "recognized_error" for item in records),
            "domains": dict(sorted(domains.items()))}


def update_error_memory(ledger: dict, association: dict, causal: dict,
                        at_curricula: int) -> dict:
    weakened = {(item.get("cue"), item.get("associated_outcome")): item
                for item in association.get("predictive_associations", [])
                if item.get("status") == "weakened"}
    for item in association.get("predictions", []):
        if item.get("correct") is not False:
            continue
        changed = [weakened[(cue, item.get("prediction"))] for cue in item.get("cues", [])
                   if (cue, item.get("prediction")) in weakened]
        _remember(ledger, domain="association", context=item.get("prior", ""),
                  asserted=item.get("prediction"), observed=item.get("observed"),
                  occurrences=item.get("count", 1),
                  evidence={"kind": "held_out_observation", "at_curricula": at_curricula,
                            "cues": item.get("cues", [])},
                  correction={"action": "weaken_association" if changed else "retain_counterexample",
                              "mechanism_changed": bool(changed),
                              "affected_links": [{"cue": link["cue"],
                                                  "outcome": link["associated_outcome"],
                                                  "successes": link["prediction_successes"],
                                                  "failures": link["prediction_failures"]}
                                                 for link in changed]})
    for item in causal.get("preregistered_predictions", []):
        if item.get("correct") is not False:
            continue
        basis = item.get("confidence_basis")
        _remember(ledger, domain="causal_candidate", context=item.get("prior", ""),
                  asserted=item.get("prediction"),
                  observed=item.get("observed_after_registration"),
                  occurrences=item.get("count", 1),
                  evidence={"kind": "preregistered_holdout", "at_curricula": at_curricula,
                            "confidence_basis": basis},
                  correction={"action": ("register_counterexample_against_hypothesis" if basis
                                         else "retain_counterexample"),
                              "mechanism_changed": False,
                              "affected_hypothesis": basis,
                              "causal_credit": False})
    ledger["summary"] = summarize(ledger)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent / ".local"
    parser.add_argument("--runtime", type=Path, default=root)
    args = parser.parse_args()
    ledger_path = args.runtime / "error-memory.json"
    ledger = (json.loads(ledger_path.read_text(encoding="utf-8"))
              if ledger_path.exists() else empty_error_memory())
    association = json.loads((args.runtime / "association-memory.json").read_text(encoding="utf-8"))
    causal = json.loads((args.runtime / "causal-memory.json").read_text(encoding="utf-8"))
    memory = json.loads((args.runtime / "global-language-memory.json").read_text(encoding="utf-8"))
    update_error_memory(ledger, association, causal, len(memory.get("merged_seeds", [])))
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, separators=(",", ":")) + "\n",
                           encoding="utf-8")
    print(json.dumps(ledger["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
