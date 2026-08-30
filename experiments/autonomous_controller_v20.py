#!/usr/bin/env python3
"""Persistent budgeted controller over AI_Noise's developmental learning layers."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from developmental_language_v15 import MultiLevelLearningAgent
from lexical_research_v16 import LexicalResearchAgent, WiktionaryDefinitions
from phrase_learning_v17 import PhraseResearchAgent
from web_cache import WEB_CACHE, NetworkBudgetExceeded


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
    stop_reason: str | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    updated_at: str | None = None


class LearningEnvironment(Protocol):
    def gaps(self) -> list[LearningGap]: ...
    def execute(self, gap: LearningGap) -> dict: ...
    def snapshot(self) -> dict: ...
    def restore(self, cycles: list[dict]) -> None: ...


class EnglishDevelopmentEnvironment:
    def __init__(self, seed: str):
        self.seed = seed
        self.agent = MultiLevelLearningAgent()
        try:
            self.bootstrap = self.agent.learn_from_web(seed)
        except NetworkBudgetExceeded as error:
            # Preserve any observations collected before the hard budget boundary.
            self.bootstrap = {"status": "network_budget_exhausted", "error": str(error)}
        self.sources = [
            WiktionaryDefinitions("en.wiktionary.org", "English Wiktionary", 0.82, True),
            WiktionaryDefinitions("simple.wiktionary.org", "Simple English Wiktionary", 0.78, False),
        ]

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
        raise ValueError(f"unsupported gap layer: {gap.layer}")

    def snapshot(self) -> dict:
        return {"bootstrap": self.bootstrap,
                "lexicon": self.agent.lexicon.report(), "story": self.agent.story.report(),
                "concepts": self.agent.concepts.ledger.report()}


class AutonomousController:
    def __init__(self, environment: LearningEnvironment, state: PersistentState, state_path: Path | None = None):
        self.environment = environment
        self.state = state
        self.state_path = state_path

    @classmethod
    def load(cls, environment: LearningEnvironment, seed: str, state_path: Path | None):
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
        return cls(environment, state, state_path)

    def save(self) -> None:
        self.state.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(asdict(self.state), ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")

    def run(self, max_steps: int, max_seconds: float) -> dict:
        started = time.monotonic()
        self.state.stop_reason = None
        for _ in range(max_steps):
            if time.monotonic() - started >= max_seconds:
                self.state.stop_reason = "time_budget_exhausted"
                break
            available = [gap for gap in self.environment.gaps()
                         if gap.gap_id not in self.state.completed_gap_ids]
            if not available:
                bootstrap = self.environment.snapshot().get("bootstrap", {})
                self.state.stop_reason = (
                    "network_budget_exhausted"
                    if bootstrap.get("status") == "network_budget_exhausted"
                    else "no_unresolved_executable_gap"
                )
                break
            selected = max(available, key=lambda gap: (gap.expected_information_gain, gap.gap_id))
            before = WEB_CACHE.stats()["network_requests"]
            try:
                result = self.environment.execute(selected)
            except NetworkBudgetExceeded:
                self.state.stop_reason = "network_budget_exhausted"
                break
            after = WEB_CACHE.stats()["network_requests"]
            self.state.completed_gap_ids.append(selected.gap_id)
            self.state.cycles.append({
                "gap": {**asdict(selected), "expected_information_gain": selected.expected_information_gain},
                "result": result, "actual_network_requests": after - before,
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
    args = parser.parse_args()
    WEB_CACHE.set_network_budget(args.max_network)
    environment = EnglishDevelopmentEnvironment(args.seed)
    controller = AutonomousController.load(environment, args.seed, args.state)
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
