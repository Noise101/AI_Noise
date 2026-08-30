#!/usr/bin/env python3
"""Small-step learning from child-level stories, without an LLM.

The learner observes simple event sequences, predicts the next event from its
own transition counts, and revises those counts when reality disagrees.  Text
understanding is deliberately tiny and inspectable: one sentence becomes one
subject/action/object event.  This is a foundation to improve, not a claim of
general language understanding.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@dataclass(frozen=True)
class Event:
    subject: str
    action: str
    object: str = ""

    @property
    def key(self) -> str:
        return "|".join((self.subject, self.action, self.object))


@dataclass
class Revision:
    context: str
    predicted: str
    observed: str
    was_correct: bool


@dataclass
class WhyQuestion:
    outcome: str
    context: str
    question: str
    status: str = "open"


class TinyStoryParser:
    """Transparent baseline parser for controlled, child-level prose."""

    ARTICLES = {"a", "an", "the"}

    def parse(self, sentence: str) -> Event | None:
        words = [word.lower() for word in WORD.findall(sentence)]
        words = [word for word in words if word not in self.ARTICLES]
        if len(words) < 2:
            return None
        return Event(words[0], words[1], "_".join(words[2:]))


class StoryLearner:
    def __init__(self):
        self.parser = TinyStoryParser()
        self.transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self.event_counts: Counter[str] = Counter()
        self.revisions: list[Revision] = []
        self.why_questions: list[WhyQuestion] = []

    def predict(self, context: Event) -> str | None:
        choices = self.transitions.get(context.key)
        if not choices:
            return None
        return max(choices, key=lambda event: (choices[event], event))

    def observe_story(self, sentences: list[str]) -> None:
        events = [event for sentence in sentences if (event := self.parser.parse(sentence))]
        for event in events:
            self.event_counts[event.key] += 1
        for before, observed in zip(events, events[1:]):
            predicted = self.predict(before)
            if predicted is not None:
                self.revisions.append(Revision(before.key, predicted, observed.key, predicted == observed.key))
                if predicted != observed.key:
                    self.why_questions.append(WhyQuestion(
                        outcome=observed.key,
                        context=before.key,
                        question=f"Why did {observed.key} happen after {before.key}?",
                    ))
            self.transitions[before.key][observed.key] += 1

    def ask_why(self, outcome: str) -> dict:
        """Return the strongest contrastive explanation, or admit uncertainty.

        Evidence is the outcome rate after a context versus its overall base
        rate.  This identifies a testable candidate cause, not proven causality.
        """
        total_transitions = sum(sum(values.values()) for values in self.transitions.values())
        outcome_total = sum(values[outcome] for values in self.transitions.values())
        if total_transitions == 0 or outcome_total == 0:
            return {"question": f"Why {outcome}?", "answer": "unknown", "reason": "no observations"}
        base_rate = outcome_total / total_transitions
        candidates = []
        for context, values in self.transitions.items():
            context_total = sum(values.values())
            support = values[outcome]
            if not support:
                continue
            conditional = support / context_total
            candidates.append((conditional - base_rate, support, conditional, context))
        lift, support, conditional, context = max(candidates)
        if support < 2 or lift <= 0:
            return {
                "question": f"Why {outcome}?", "answer": "unknown",
                "reason": "no contrastive evidence yet", "best_candidate": context,
            }
        return {
            "question": f"Why {outcome}?",
            "answer": f"possibly because {context}",
            "candidate_cause": context,
            "confidence": round(conditional, 3),
            "lift_over_baseline": round(lift, 3),
            "support": support,
            "warning": "predictive evidence, not yet a proven cause",
        }

    def confidence(self, context: str, outcome: str) -> float:
        counts = self.transitions.get(context, Counter())
        total = sum(counts.values())
        return 0.0 if total == 0 else counts[outcome] / total

    def report(self) -> dict:
        learned_rules = []
        for context, outcomes in sorted(self.transitions.items()):
            total = sum(outcomes.values())
            for outcome, count in outcomes.most_common():
                learned_rules.append({
                    "when": context,
                    "expect": outcome,
                    "observations": count,
                    "confidence": round(count / total, 3),
                })
        mistakes = [asdict(item) for item in self.revisions if not item.was_correct]
        why_answers = []
        for question in self.why_questions:
            answer = self.ask_why(question.outcome)
            question.status = "candidate_found" if answer.get("candidate_cause") else "open"
            why_answers.append({**asdict(question), "current_answer": answer})
        return {
            "method": "counted event transitions; no pretrained model or LLM",
            "events_seen": sum(self.event_counts.values()),
            "rules": learned_rules,
            "predictions_checked": len(self.revisions),
            "mistakes_detected": len(mistakes),
            "revisions": mistakes,
            "why_questions": why_answers,
        }


DEMO_STORIES = [
    ["Fox sees grapes.", "Fox jumps high.", "Fox misses grapes."],
    ["Fox sees grapes.", "Fox jumps high.", "Fox misses grapes."],
    ["Fox sees grapes.", "Fox waits quietly.", "Bird drops grapes."],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    learner = StoryLearner()
    for story in DEMO_STORIES:
        learner.observe_story(story)
    rendered = json.dumps(learner.report(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
