"""Evidence-based self-assessment of the learner's own language abilities."""

from __future__ import annotations


TARGET = 0.85


def ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(min(1.0, numerator / denominator), 3)


def assess_language_mastery(report: dict, causal_evaluation: dict | None = None,
                            conversation_practice: dict | None = None,
                            representation_evaluation: dict | None = None) -> dict:
    knowledge = report.get("knowledge", {})
    lexicon = knowledge.get("lexicon", {})
    story = knowledge.get("story", {})
    concepts = knowledge.get("concepts", {})

    characters = lexicon.get("characters", {})
    stable_characters = sum(count >= 2 for count in characters.values())
    words = lexicon.get("word_forms", {})
    grounded_words = {item.get("form") for item in lexicon.get("grounded_meanings", [])}
    researched_words = {form for form, belief in lexicon.get("researched_meanings", {}).items()
                        if belief.get("accepted_sense")}
    phrases = lexicon.get("phrase_candidates", [])
    researched_phrases = sum(bool(item.get("accepted_sense")) for item in
                             lexicon.get("researched_phrase_meanings", {}).values())
    cues = lexicon.get("conversation_cues", {})
    grounded_cues = sum(bool(item.get("accepted_sense")) for item in
                        lexicon.get("researched_conversation_acts", {}).values())
    causal_evaluation = causal_evaluation or {}
    evaluation = causal_evaluation.get("evaluation", {})
    representation_evaluation = representation_evaluation or {}
    predictive = representation_evaluation.get("selected_evaluation", {})
    predictions = predictive.get("total", 0)
    correct = predictive.get("correct", 0)
    predictive_baseline = predictive.get("baseline_correct", 0)
    causal_total = evaluation.get("total", 0)
    causal_correct = evaluation.get("correct", 0)
    baseline_correct = evaluation.get("baseline_correct", 0)
    supported = causal_evaluation.get("supported_hypotheses", 0)
    beliefs = concepts.get("beliefs", [])
    corroborated = sum(item.get("status") == "corroborated" for item in beliefs)

    conversation_practice = conversation_practice or {}
    practice_turns = conversation_practice.get("evaluated_turns", 0)
    practice_successes = conversation_practice.get("successful_followups", 0)
    dimensions = {
        "characters": {"score": ratio(stable_characters, max(10, len(characters))),
                       "evidence": f"{stable_characters}/{len(characters)} characters repeated"},
        "words": {"score": ratio(len(grounded_words | researched_words), max(1, len(words))),
                  "evidence": f"{len(grounded_words | researched_words)}/{len(words)} forms grounded"},
        "phrases": {"score": ratio(researched_phrases, max(1, len(phrases))),
                    "evidence": f"{researched_phrases}/{len(phrases)} repeated phrases grounded"},
        "conversation": {"score": (0.0 if practice_turns < 3 else ratio(practice_successes, practice_turns)),
                         "evidence": f"{practice_successes}/{practice_turns} self-generated followups usable; partner claims excluded"},
        "prediction": {"score": (0.0 if predictions < 20 or correct <= predictive_baseline
                                  else ratio(correct, predictions)),
                       "evidence": f"{predictions} representation holdout checks, {correct} correct versus baseline {predictive_baseline}"},
        "causality": {"score": (0.0 if causal_total < 20 or causal_correct <= baseline_correct
                                  else ratio(supported, max(5, supported))),
                     "evidence": f"{supported} candidates; {causal_correct}/{causal_total} versus baseline {baseline_correct}/{causal_total}"},
        "concepts": {"score": ratio(corroborated, max(3, len(beliefs))),
                    "evidence": f"{corroborated}/{len(beliefs)} beliefs corroborated"},
    }
    weakest_name, weakest = min(dimensions.items(), key=lambda item: (item[1]["score"], item[0]))
    overall = round(sum(item["score"] for item in dimensions.values()) / len(dimensions), 3)
    return {
        "status": "current_curriculum_mastered" if all(item["score"] >= TARGET for item in dimensions.values())
                  else "learning_incomplete",
        "overall_score": overall,
        "target": TARGET,
        "dimensions": dimensions,
        "weakest_dimension": weakest_name,
        "next_mastery_goal": {
            "dimension": weakest_name,
            "reason": weakest["evidence"],
            "objective": f"seek new observations that test and ground {weakest_name}",
        },
        "warning": "mastery is bounded to observed curricula; it is never a claim of complete language ability",
    }
