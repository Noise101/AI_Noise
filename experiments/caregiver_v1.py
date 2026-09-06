#!/usr/bin/env python3
"""Low-effort caregiver check for the developmental Japanese reading loop.

Every CAREGIVER_INTERVAL cycles Noise writes a small batch of
multiple-choice / yes-no questions about books it has just read.  Answering
is one short command -- positional digits or はい/いいえ, e.g.

    python3 japanese_reader_v1.py answer "2 はい 1"

The point is a human-verified signal: for each answered question we record
whether the caregiver's answer matches (a) the story's actual structure and
(b) what Noise's comprehension model predicted.  That gives a labelled
accuracy the automated frozen benchmark cannot: does Noise agree with a
person about what happened?

Unanswered questions are harmless -- they expire on the next batch and the
reading loop keeps running on its automated evaluation.
"""

from __future__ import annotations

import hashlib
from collections import Counter

CAREGIVER_INTERVAL = 15          # cycles between question batches
QUESTIONS_PER_BATCH = 3
QUESTION_TTL_CYCLES = 40         # a batch expires if left unanswered this long


def empty_state() -> dict:
    return {"version": 1, "last_batch_cycle": 0, "pending": [], "answered": [],
            "human_checked": 0, "human_matches_story": 0, "human_matches_model": 0}


def due(state: dict, cycle: int) -> bool:
    if state.get("pending"):
        return False
    return cycle - state.get("last_batch_cycle", 0) >= CAREGIVER_INTERVAL


def _qid(book_url: str, cycle: int, n: int) -> str:
    return hashlib.sha256(f"{book_url}:{cycle}:{n}".encode()).hexdigest()[:10]


def _distinct(seq):
    return list(dict.fromkeys(x for x in seq if x))


_PARTICLE_CHARS = set("はがをにへでとのも、。")


def _name_like(subject: str) -> bool:
    """A caregiver question is only worth asking if the candidate answers look
    like real story entities, not parser debris ('冬はあつぼったい木のくつを')."""
    return bool(subject) and 2 <= len(subject) <= 6 and not (_PARTICLE_CHARS & set(subject))


def generate_questions(recent: list[dict], cycle: int, model=None) -> list[dict]:
    """recent: [{"title", "url", "events"}], most-recently-read first.

    Question kinds, all gradeable without parsing free text:
      * protagonist -- who is the story about (3 options)
      * actor       -- who did <obj> <verb> (options)
      * order       -- did <verb A> happen before <verb B> (はい/いいえ)
    """
    questions: list[dict] = []
    for book in recent:
        if len(questions) >= QUESTIONS_PER_BATCH:
            break
        events = [e for e in book.get("events", []) if e.get("verb")]
        if len(events) < 3:
            continue
        subjects = [s for s in _distinct(e.get("subject") for e in events) if _name_like(s)]
        title = book.get("title") or "この本"
        url = book.get("url") or title
        n = len(questions)

        if len(subjects) >= 2:
            counts = Counter(e.get("subject") for e in events if _name_like(e.get("subject") or ""))
            protagonist = counts.most_common(1)[0][0]
            options = _distinct([protagonist] + subjects)[:3]
            options_sorted = sorted(options, key=lambda o: hashlib.md5(f"{url}{o}".encode()).hexdigest())
            questions.append({
                "id": _qid(url, cycle, n), "cycle": cycle, "book": title, "url": url,
                "kind": "protagonist",
                "prompt": f"「{title}」は だれの お話ですか？　"
                          + "　".join(f"{i+1}) {o}" for i, o in enumerate(options_sorted)),
                "options": options_sorted,
                "answer_story": options_sorted.index(protagonist) + 1,
                "answer_model": (options_sorted.index(model.predict_protagonist(events[:2], options_sorted)) + 1)
                                if model is not None else None,
            })
            continue

        verbs = _distinct(e.get("verb") for e in events)
        if len(verbs) >= 2:
            a, b = verbs[0], verbs[-1]
            questions.append({
                "id": _qid(url, cycle, n), "cycle": cycle, "book": title, "url": url,
                "kind": "order",
                "prompt": f"「{title}」で、「{a}」は「{b}」より さきに おきましたか？（はい/いいえ）",
                "options": ["はい", "いいえ"],
                "answer_story": 1,   # a precedes b by construction (verbs[0] .. verbs[-1])
                "answer_model": None,
            })
    return questions


def open_batch(state: dict, questions: list[dict], cycle: int) -> dict:
    state["pending"] = questions
    state["last_batch_cycle"] = cycle
    return state


def expire_if_stale(state: dict, cycle: int) -> bool:
    if state.get("pending") and cycle - state.get("last_batch_cycle", 0) > QUESTION_TTL_CYCLES:
        state["pending"] = []
        return True
    return False


_YES = {"はい", "y", "yes", "1", "うん", "そう"}
_NO = {"いいえ", "n", "no", "2", "ちがう"}


def _match(question: dict, token: str) -> tuple[bool, bool]:
    """(matches_story, is_gradeable) for one raw answer token."""
    token = token.strip()
    if not token:
        return False, False
    if question["kind"] == "order":
        if token in _YES:
            return question["answer_story"] == 1, True
        if token in _NO:
            return question["answer_story"] == 2, True
        return False, False
    # multiple choice: a digit, or the option text
    choice = None
    if token.isdigit():
        choice = int(token)
    else:
        for i, opt in enumerate(question["options"]):
            if token == opt or token in opt:
                choice = i + 1
                break
    if choice is None or not (1 <= choice <= len(question["options"])):
        return False, False
    return choice == question["answer_story"], True


def apply_answers(state: dict, raw: str) -> dict:
    """Positional: the k-th token answers the k-th pending question.  Extra
    tokens and blanks are ignored; missing answers just stay unanswered."""
    pending = state.get("pending", [])
    tokens = raw.replace(",", " ").split()
    graded = 0
    matched_story = matched_model = 0
    still_pending = []
    for i, question in enumerate(pending):
        token = tokens[i] if i < len(tokens) else ""
        ok_story, gradeable = _match(question, token)
        if not gradeable:
            still_pending.append(question)
            continue
        graded += 1
        matched_story += int(ok_story)
        model_ans = question.get("answer_model")
        ok_model = model_ans is not None and token.isdigit() and int(token) == model_ans
        matched_model += int(ok_model)
        state["answered"].append({"id": question.get("id"), "book": question.get("book"),
                                  "kind": question.get("kind"), "answer": token,
                                  "matched_story": ok_story, "matched_model": ok_model,
                                  "cycle": question.get("cycle")})
    state["answered"] = state["answered"][-500:]
    state["pending"] = still_pending
    state["human_checked"] = state.get("human_checked", 0) + graded
    state["human_matches_story"] = state.get("human_matches_story", 0) + matched_story
    state["human_matches_model"] = state.get("human_matches_model", 0) + matched_model
    return {"graded": graded, "matched_story": matched_story, "matched_model": matched_model,
            "still_pending": len(still_pending)}


def summary(state: dict) -> dict:
    checked = state.get("human_checked", 0)
    return {
        "pending": len(state.get("pending", [])),
        "human_checked": checked,
        "human_story_agreement": round(state.get("human_matches_story", 0) / checked, 3) if checked else None,
        "human_model_agreement": round(state.get("human_matches_model", 0) / checked, 3) if checked else None,
        "last_batch_cycle": state.get("last_batch_cycle", 0),
    }
