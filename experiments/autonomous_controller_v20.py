#!/usr/bin/env python3
"""Persistent budgeted controller over AI_Noise's developmental learning layers."""

from __future__ import annotations

import argparse
import json
import math
import time
import re
import urllib.error
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from developmental_language_v15 import MultiLevelLearningAgent
from lexical_research_v16 import LexicalResearchAgent, WiktionaryDefinitions
from phrase_learning_v17 import PhraseResearchAgent
from curiosity_drive_v23 import observe_unknown, resolve_unknown, result_is_grounded
from web_cache import WEB_CACHE, NetworkBudgetExceeded
from japanese_sense_grounding_v19 import run as run_japanese_learning

JAPANESE_TEXT = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


@dataclass
class LearningGap:
    gap_id: str
    layer: str
    query: str
    uncertainty: float
    observations: int
    estimated_requests: int
    reason: str

    @property
    def expected_information_gain(self) -> float:
        experience = 1 + math.log2(max(1, self.observations))
        return round(self.uncertainty * experience / max(1, self.estimated_requests), 4)


@dataclass
class PersistentState:
    seed: str
    completed_gap_ids: list[str] = field(default_factory=list)
    cycles: list[dict] = field(default_factory=list)
    curiosity_ledger: dict[str, dict] = field(default_factory=dict)
    stop_reason: str | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    updated_at: str | None = None


class LearningEnvironment(Protocol):
    def gaps(self) -> list[LearningGap]: ...
    def execute(self, gap: LearningGap) -> dict: ...
    def snapshot(self) -> dict: ...
    def restore(self, cycles: list[dict]) -> None: ...


class EnglishDevelopmentEnvironment:
    def __init__(self, seed: str, global_memory: dict | None = None):
        self.seed = seed
        self.agent = MultiLevelLearningAgent()
        self.apply_global_memory(global_memory or {})
        self.global_transition_baseline = {
            context: dict(outcomes) for context, outcomes in self.agent.story.transitions.items()}
        self.global_event_baseline = sum(self.agent.story.event_counts.values())
        try:
            self.bootstrap = self.agent.learn_from_web(seed)
        except urllib.error.HTTPError as error:
            if error.code not in {403, 404, 410}:
                raise
            self.bootstrap = {"status": "permanent_source_unavailable",
                              "http_status": error.code, "url": error.url,
                              "reason": "source denied or removed; defer this curriculum"}
        except NetworkBudgetExceeded as error:
            # Preserve any observations collected before the hard budget boundary.
            self.bootstrap = {"status": "network_budget_exhausted", "error": str(error)}
        parallel = self.bootstrap.get("parallel_learning", {})
        if parallel:
            parallel["event_predictions"] = self.local_story_report()
        self.sources = [
            WiktionaryDefinitions("en.wiktionary.org", "English Wiktionary", 0.82, True),
            WiktionaryDefinitions("simple.wiktionary.org", "Simple English Wiktionary", 0.78, False),
        ]

    def apply_global_memory(self, memory: dict) -> None:
        for form, item in memory.get("words", {}).items():
            belief = item.get("accepted_belief")
            if belief:
                self.agent.lexicon.update_meaning_hypothesis(form, belief)
        for phrase, item in memory.get("phrases", {}).items():
            belief = item.get("accepted_belief")
            if belief:
                self.agent.lexicon.update_phrase_hypothesis(
                    phrase, belief, belief.get("compositionality", {}))
        for cue, item in memory.get("conversation_acts", {}).items():
            belief = item.get("accepted_belief")
            if belief:
                self.agent.lexicon.update_conversation_hypothesis(cue, belief)
        for context, outcomes in memory.get("event_transitions", {}).items():
            self.agent.story.transitions[context].update(outcomes)
        self.agent.story.event_counts.update(memory.get("event_counts", {}))

    def restore(self, cycles: list[dict]) -> None:
        """Replay accepted learning updates, not merely completed task IDs."""
        for cycle in cycles:
            gap = cycle.get("gap", {})
            result = cycle.get("result", {})
            belief = result.get("meaning_belief")
            if not belief:
                continue
            form = gap.get("gap_id", "").split(":", 1)[-1]
            if gap.get("layer") == "word":
                self.agent.lexicon.update_meaning_hypothesis(form, belief)
            elif gap.get("layer") == "phrase":
                self.agent.lexicon.update_phrase_hypothesis(
                    form, belief, result.get("compositionality", {}))
            elif gap.get("layer") == "conversation":
                self.agent.lexicon.update_conversation_hypothesis(form, belief)

    def gaps(self) -> list[LearningGap]:
        gaps = []
        lexical = self.agent.lexicon.lexical_gap()
        if lexical:
            gaps.append(LearningGap(
                f"word:{lexical['form']}", "word", lexical["query"], 1.0,
                lexical.get("observations", 1), 2, "observed form has no accepted sourced sense",
            ))
        phrase = self.agent.lexicon.phrase_gap()
        if phrase:
            gaps.append(LearningGap(
                f"phrase:{phrase['form']}", "phrase", phrase["query"], 0.75,
                phrase.get("observations", 1), 2, "repeated phrase has no accepted phrase sense",
            ))
        conversation = self.agent.lexicon.conversation_gap()
        if conversation:
            gaps.append(LearningGap(
                f"conversation:{conversation['form']}", "conversation", conversation["query"], 1.0,
                conversation.get("observations", 1), 2,
                "observed dialogue cue has no grounded conversational function",
            ))
        for question in self.agent.story.why_questions:
            if question.status == "open":
                gaps.append(LearningGap(
                    f"why:{question.context}->{question.outcome}", "why", question.question,
                    1.0, 1, 0, "prediction failed and its cause is unexplained",
                ))
        return gaps

    def execute(self, gap: LearningGap) -> dict:
        if gap.layer == "word":
            lexical = self.agent.lexicon.lexical_gap()
            research = LexicalResearchAgent(self.sources).investigate(lexical, self.agent.lexicon.word_links)
            self.agent.lexicon.update_meaning_hypothesis(lexical["form"], research["meaning_belief"])
            return research
        if gap.layer == "phrase":
            phrase = self.agent.lexicon.phrase_gap()
            research = PhraseResearchAgent(self.sources).investigate(phrase, self.agent.lexicon.meaning_hypotheses)
            self.agent.lexicon.update_phrase_hypothesis(
                phrase["form"], research["meaning_belief"], research["compositionality"])
            return research
        if gap.layer == "why":
            outcome = gap.gap_id.split("->", 1)[1]
            return {"why": self.agent.story.ask_why(outcome),
                    "investigation": self.agent.story.plan_why_investigation(outcome)}
        if gap.layer == "conversation":
            cue = gap.gap_id.split(":", 1)[1]
            lexical = {"form": cue, "query": gap.query,
                       "contexts": list(self.agent.lexicon.conversation_contexts[cue]),
                       "observations": self.agent.lexicon.conversation_cues[cue]}
            research = LexicalResearchAgent(self.sources).investigate(lexical, self.agent.lexicon.word_links)
            belief = research["meaning_belief"]
            self.agent.lexicon.update_conversation_hypothesis(cue, belief)
            return {**research, "grounded_pattern": belief.get("accepted_sense")}
        raise ValueError(f"unsupported gap layer: {gap.layer}")

    def snapshot(self) -> dict:
        return {"bootstrap": self.bootstrap,
                "lexicon": self.agent.lexicon.report(), "story": self.local_story_report(),
                "concepts": self.agent.concepts.ledger.report()}

    def local_story_report(self) -> dict:
        report = self.agent.story.report()
        local_rules = []
        for rule in report.get("rules", []):
            baseline = self.global_transition_baseline.get(rule["when"], {}).get(rule["expect"], 0)
            local_count = rule["observations"] - baseline
            if local_count > 0:
                local_rules.append({**rule, "observations": local_count})
        report["rules"] = local_rules
        report["events_seen"] = max(0, report.get("events_seen", 0) - self.global_event_baseline)
        report["global_prior"] = {"transition_contexts": len(self.global_transition_baseline),
                                  "events": self.global_event_baseline,
                                  "included_in_local_rules": False}
        return report


class JapaneseDevelopmentEnvironment:
    """Adapter that puts the existing Japanese self-learning path under the same controller."""

    def __init__(self, seed: str, global_memory: dict | None = None):
        self.seed = seed
        try:
            self.result = run_japanese_learning(seed, use_local_helper=False)
        except urllib.error.HTTPError as error:
            if error.code not in {403, 404, 410}:
                raise
            self.result = {"status": "permanent_source_unavailable",
                           "http_status": error.code, "url": error.url}
        except NetworkBudgetExceeded as error:
            self.result = {"status": "network_budget_exhausted", "error": str(error)}

    def restore(self, cycles: list[dict]) -> None:
        return None

    def gaps(self) -> list[LearningGap]:
        return []

    def execute(self, gap: LearningGap) -> dict:
        raise ValueError("Japanese observations are acquired during boundary/sense bootstrap")

    def snapshot(self) -> dict:
        boundary = self.result.get("boundary_learning", {})
        accepted = boundary.get("accepted_words", [])
        grounded = self.result.get("sense_grounding") or {}
        belief = grounded.get("belief", {})
        form = grounded.get("surface")
        researched = ({form: belief} if form and belief.get("accepted_sense") else {})
        return {"bootstrap": self.result, "lexicon": {
            "characters": {}, "word_forms": {item["form"]: item.get("count", 1) for item in accepted},
            "grounded_meanings": [{"form": item["form"]} for item in accepted],
            "researched_meanings": researched, "phrase_candidates": [],
            "researched_phrase_meanings": {}, "conversation_cues": {},
            "researched_conversation_acts": {}},
            "story": {"predictions_checked": 0, "mistakes_detected": 0,
                      "why_questions": [], "rules": [], "events_seen": 0},
            "concepts": {"beliefs": []}}


class AutonomousController:
    def __init__(self, environment: LearningEnvironment, state: PersistentState,
                 state_path: Path | None = None, curiosity_priors: dict | None = None):
        self.environment = environment
        self.state = state
        self.state_path = state_path
        self.curiosity_priors = curiosity_priors or {}

    @classmethod
    def load(cls, environment: LearningEnvironment, seed: str, state_path: Path | None,
             curiosity_priors: dict | None = None):
        if state_path and state_path.exists():
            data = json.loads(state_path.read_text(encoding="utf-8"))
            if data.get("seed") != seed:
                raise ValueError("state seed does not match requested seed")
            state = PersistentState(**data)
        else:
            state = PersistentState(seed)
        restore = getattr(environment, "restore", None)
        if restore:
            restore(state.cycles)
        return cls(environment, state, state_path, curiosity_priors)

    def save(self) -> None:
        self.state.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(asdict(self.state), ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")

    def run(self, max_steps: int, max_seconds: float) -> dict:
        started = time.monotonic()
        attempted_this_run: set[str] = set()
        self.state.stop_reason = None
        for _ in range(max_steps):
            if time.monotonic() - started >= max_seconds:
                self.state.stop_reason = "time_budget_exhausted"
                break
            gaps = self.environment.gaps()
            cycle_number = len(self.state.cycles)
            for gap in gaps:
                entry = observe_unknown(self.state.curiosity_ledger, gap, cycle_number)
                prior = self.curiosity_priors.get(gap.gap_id, {})
                entry["global_pressure_prior"] = prior.get("pressure", 0.0)
            available = [gap for gap in gaps
                         if gap.gap_id not in self.state.completed_gap_ids
                         and self.state.curiosity_ledger[gap.gap_id].get("last_attempt_encounters", -1)
                         < gap.observations
                         and gap.gap_id not in attempted_this_run]
            if not available:
                bootstrap = self.environment.snapshot().get("bootstrap", {})
                self.state.stop_reason = (
                    "network_budget_exhausted"
                    if bootstrap.get("status") == "network_budget_exhausted"
                    else ("no_new_evidence_for_unresolved_gap" if attempted_this_run
                          else "no_unresolved_executable_gap")
                )
                break
            selected = max(available, key=lambda gap: (
                self.state.curiosity_ledger[gap.gap_id]["pressure"]
                + self.state.curiosity_ledger[gap.gap_id].get("global_pressure_prior", 0.0),
                gap.expected_information_gain, gap.gap_id))
            selected_pressure = (self.state.curiosity_ledger[selected.gap_id]["pressure"]
                                 + self.state.curiosity_ledger[selected.gap_id].get(
                                     "global_pressure_prior", 0.0))
            before = WEB_CACHE.stats()["network_requests"]
            try:
                result = self.environment.execute(selected)
            except NetworkBudgetExceeded:
                self.state.stop_reason = "network_budget_exhausted"
                break
            after = WEB_CACHE.stats()["network_requests"]
            grounded, resolution = result_is_grounded(selected.layer, result)
            attempted_this_run.add(selected.gap_id)
            self.state.curiosity_ledger[selected.gap_id]["last_attempt_encounters"] = selected.observations
            if grounded:
                self.state.completed_gap_ids.append(selected.gap_id)
                resolve_unknown(self.state.curiosity_ledger, selected.gap_id, resolution, cycle_number + 1)
            self.state.cycles.append({
                "gap": {**asdict(selected), "expected_information_gain": selected.expected_information_gain,
                        "curiosity_pressure": selected_pressure},
                "result": result, "actual_network_requests": after - before,
                "grounded": grounded,
            })
            self.save()
        else:
            self.state.stop_reason = "step_budget_exhausted"
        self.save()
        return self.report(started)

    def report(self, started: float) -> dict:
        return {"state": asdict(self.state), "current_gaps": [
                    {**asdict(gap), "expected_information_gain": gap.expected_information_gain}
                    for gap in self.environment.gaps() if gap.gap_id not in self.state.completed_gap_ids],
                "knowledge": self.environment.snapshot(), "web_usage": WEB_CACHE.stats(),
                "elapsed_seconds": round(time.monotonic() - started, 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", nargs="?", default="fox grapes")
    parser.add_argument("--state", type=Path, default=Path("controller-state.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--max-network", type=int, default=8)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--curiosity-priors", type=Path)
    parser.add_argument("--global-memory", type=Path)
    args = parser.parse_args()
    WEB_CACHE.set_network_budget(args.max_network)
    global_memory = {}
    if args.global_memory and args.global_memory.exists():
        global_memory = json.loads(args.global_memory.read_text(encoding="utf-8"))
    environment = (JapaneseDevelopmentEnvironment(args.seed, global_memory)
                   if JAPANESE_TEXT.search(args.seed)
                   else EnglishDevelopmentEnvironment(args.seed, global_memory))
    priors = {}
    if args.curiosity_priors and args.curiosity_priors.exists():
        priors = json.loads(args.curiosity_priors.read_text(encoding="utf-8"))
    controller = AutonomousController.load(environment, args.seed, args.state, priors)
    report = controller.run(args.max_steps, args.max_seconds)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.summary:
        print(json.dumps({"stop_reason": report["state"]["stop_reason"],
                          "completed": len(report["state"]["completed_gap_ids"]),
                          "remaining_gaps": len(report["current_gaps"]),
                          "web_usage": report["web_usage"],
                          "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))
    else:
        print(rendered)


if __name__ == "__main__":
    main()
