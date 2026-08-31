#!/usr/bin/env python3
"""Canonical cross-curriculum language memory reconstructed from seed reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def empty_memory() -> dict:
    return {"version": 27, "merged_seeds": [], "characters": {}, "words": {}, "phrases": {},
            "conversation_acts": {}, "event_transitions": {}, "event_counts": {},
            "quality_event_transitions": {}, "quality_event_counts": {},
            "prediction": {"checked": 0, "mistakes": 0, "why_gaps": 0, "why_candidates": 0},
            "concepts": {}, "totals": {}}


def merge_report(memory: dict, seed: str, report: dict) -> bool:
    memory.setdefault("quality_event_transitions", {})
    memory.setdefault("quality_event_counts", {})
    is_new_seed = seed not in set(memory.get("merged_seeds", []))
    knowledge = report.get("knowledge", {})
    lexicon = knowledge.get("lexicon", {})
    for character, count in lexicon.get("characters", {}).items():
        memory["characters"][character] = memory["characters"].get(character, 0) + count
    roles = {item.get("form"): item for item in lexicon.get("grounded_meanings", [])}
    researched = lexicon.get("researched_meanings", {})
    for form, count in lexicon.get("word_forms", {}).items():
        entry = memory["words"].setdefault(form, {"encounters": 0, "curricula": 0,
                                                   "roles": {}, "accepted_belief": None})
        if is_new_seed:
            entry["encounters"] += count
            entry["curricula"] += 1
            for role, role_count in roles.get(form, {}).get("roles", {}).items():
                entry["roles"][role] = entry["roles"].get(role, 0) + role_count
        belief = researched.get(form)
        if belief and belief.get("accepted_sense"):
            entry["accepted_belief"] = belief
    for candidate in lexicon.get("phrase_candidates", []):
        phrase = candidate.get("phrase")
        if not phrase:
            continue
        entry = memory["phrases"].setdefault(phrase, {"encounters": 0, "curricula": 0,
                                                       "accepted_belief": None})
        if is_new_seed:
            entry["encounters"] += candidate.get("count", 0)
            entry["curricula"] += 1
        belief = lexicon.get("researched_phrase_meanings", {}).get(phrase)
        if belief and belief.get("accepted_sense"):
            entry["accepted_belief"] = belief
    for cue, count in lexicon.get("conversation_cues", {}).items():
        entry = memory["conversation_acts"].setdefault(
            cue, {"encounters": 0, "curricula": 0, "accepted_belief": None})
        if is_new_seed:
            entry["encounters"] += count
            entry["curricula"] += 1
        belief = lexicon.get("researched_conversation_acts", {}).get(cue)
        if belief and belief.get("accepted_sense"):
            entry["accepted_belief"] = belief
    if is_new_seed:
        story = knowledge.get("story", {})
        memory["prediction"]["checked"] += story.get("predictions_checked", 0)
        memory["prediction"]["mistakes"] += story.get("mistakes_detected", 0)
        memory["prediction"]["why_gaps"] += len(story.get("why_questions", []))
        memory["prediction"]["why_candidates"] += sum(
            item.get("status") == "candidate_found" for item in story.get("why_questions", []))
        for rule in story.get("rules", []):
            context, outcome = rule.get("when"), rule.get("expect")
            if context and outcome:
                outcomes = memory["event_transitions"].setdefault(context, {})
                outcomes[outcome] = outcomes.get(outcome, 0) + rule.get("observations", 0)
        for source in knowledge.get("bootstrap", {}).get("sources", []):
            for event in source.get("learned_events", []):
                memory["event_counts"][event] = memory["event_counts"].get(event, 0) + 1
            previous = None
            for item in source.get("event_extraction_audit", []):
                event = item.get("event") if item.get("accepted") else None
                if not event:
                    previous = None
                    continue
                memory["quality_event_counts"][event] = memory["quality_event_counts"].get(event, 0) + 1
                if previous:
                    outcomes = memory["quality_event_transitions"].setdefault(previous, {})
                    outcomes[event] = outcomes.get(event, 0) + 1
                previous = event
        for belief in knowledge.get("concepts", {}).get("beliefs", []):
            key = "|".join(str(belief.get(name, "")) for name in ("subject", "relation", "object", "scope"))
            entry = memory["concepts"].setdefault(key, {"observations": 0, "curricula": 0,
                                                        "statuses": {}, "citations": []})
            entry["observations"] += 1
            entry["curricula"] += 1
            status = belief.get("status", "unknown")
            entry["statuses"][status] = entry["statuses"].get(status, 0) + 1
            entry["citations"] = sorted(set(entry["citations"]) | set(belief.get("citations", [])))
        memory["merged_seeds"].append(seed)
    memory["totals"] = summarize(memory)
    return is_new_seed


def summarize(memory: dict) -> dict:
    return {"curricula": len(memory.get("merged_seeds", [])),
            "word_forms": len(memory.get("words", {})),
            "grounded_word_forms": sum(bool(item.get("roles") or item.get("accepted_belief"))
                                       for item in memory.get("words", {}).values()),
            "phrases": len(memory.get("phrases", {})),
            "grounded_phrases": sum(bool(item.get("accepted_belief"))
                                    for item in memory.get("phrases", {}).values()),
            "conversation_acts": len(memory.get("conversation_acts", {})),
            "grounded_conversation_acts": sum(bool(item.get("accepted_belief"))
                                              for item in memory.get("conversation_acts", {}).values()),
            "event_contexts": len(memory.get("event_transitions", {})),
            "events": len(memory.get("event_counts", {})),
            "concepts": len(memory.get("concepts", {}))}


def mastery_report(memory: dict) -> dict:
    words = memory.get("words", {})
    phrases = memory.get("phrases", {})
    conversations = memory.get("conversation_acts", {})
    concept_beliefs = [{"status": max(item.get("statuses", {"unknown": 0}),
                                       key=item.get("statuses", {"unknown": 0}).get)}
                       for item in memory.get("concepts", {}).values()]
    prediction = memory.get("prediction", {})
    return {"knowledge": {"lexicon": {
        "characters": memory.get("characters", {}),
        "word_forms": {key: item["encounters"] for key, item in words.items()},
        "grounded_meanings": [{"form": key} for key, item in words.items() if item.get("roles")],
        "researched_meanings": {key: item["accepted_belief"] for key, item in words.items()
                                if item.get("accepted_belief")},
        "phrase_candidates": [{"phrase": key} for key in phrases],
        "researched_phrase_meanings": {key: item["accepted_belief"] for key, item in phrases.items()
                                       if item.get("accepted_belief")},
        "conversation_cues": {key: item["encounters"] for key, item in conversations.items()},
        "researched_conversation_acts": {key: item["accepted_belief"]
                                         for key, item in conversations.items()
                                         if item.get("accepted_belief")},
    }, "story": {"predictions_checked": prediction.get("checked", 0),
                  "mistakes_detected": prediction.get("mistakes", 0),
                  "why_questions": ([{"status": "candidate_found"}] * prediction.get("why_candidates", 0)
                                    + [{"status": "open"}] * max(0, prediction.get("why_gaps", 0)
                                                                  - prediction.get("why_candidates", 0)))},
        "concepts": {"beliefs": concept_beliefs}}}


def rebuild(runtime: Path) -> dict:
    memory = empty_memory()
    report_paths = [runtime / "latest-report.json", *sorted((runtime / "seeds").glob("*/latest-report.json"))]
    for path in report_paths:
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        seed = report.get("state", {}).get("seed")
        if seed:
            merge_report(memory, seed, report)
    return memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=Path(__file__).resolve().parent.parent / ".local")
    args = parser.parse_args()
    memory = rebuild(args.runtime)
    output = args.runtime / "global-language-memory.json"
    output.write_text(json.dumps(memory, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **memory["totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
