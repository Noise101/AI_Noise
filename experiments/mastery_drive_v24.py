"""Evidence-based self-assessment of the learner's own language abilities."""

from __future__ import annotations


TARGET = 0.85


def ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(min(1.0, numerator / denominator), 3)


def assess_language_mastery(report: dict) -> dict:
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
    predictions = story.get("predictions_checked", 0)
    mistakes = story.get("mistakes_detected", 0)
    why = story.get("why_questions", [])
    explained = sum(item.get("status") == "candidate_found" for item in why)
    beliefs = concepts.get("beliefs", [])
    corroborated = sum(item.get("status") == "corroborated" for item in beliefs)

    dimensions = {
        "characters": {"score": ratio(stable_characters, max(10, len(characters))),
                       "evidence": f"{stable_characters}/{len(characters)} characters repeated"},
        "words": {"score": ratio(len(grounded_words | researched_words), max(1, len(words))),
                  "evidence": f"{len(grounded_words | researched_words)}/{len(words)} forms grounded"},
        "phrases": {"score": ratio(researched_phrases, max(1, len(phrases))),
                    "evidence": f"{researched_phrases}/{len(phrases)} repeated phrases grounded"},
        "conversation": {"score": ratio(grounded_cues, max(1, len(cues))),
                         "evidence": f"{grounded_cues}/{len(cues)} dialogue acts grounded"},
        "prediction": {"score": (0.0 if predictions < 3 else ratio(predictions - mistakes, predictions)),
                       "evidence": f"{predictions} checked, {mistakes} mistakes"},
        "causality": {"score": ratio(explained, max(1, len(why))),
                     "evidence": f"{explained}/{len(why)} why gaps have candidates"},
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
