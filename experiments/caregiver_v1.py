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


_PARTICLE_CHARS = set("はがをへ、。")          # rare inside a noun (もり/こうもり/つの keep も/の)
# pronouns / deixis / quantifiers / connectives / adverbs the parser leaves as
# "subjects" -- never a meaningful multiple-choice answer for "whose story is it?"
_NON_ENTITY = {"それ", "これ", "あれ", "どれ", "ここ", "そこ", "あそこ", "わたし", "わたくし",
               "あなた", "きみ", "おまえ", "ぼく", "おれ", "だれ", "なに", "みんな", "みな",
               "ひとり", "ふたり", "なるほう", "ある", "いる", "こと", "もの", "とき", "ところ",
               "じぶん", "ひとつ", "そう", "どう", "なるほど",
               "けれど", "けれども", "しかし", "そして", "それから", "すると", "ところが",
               "いつか", "いつも", "いきなり", "やがて", "とうとう", "なぜ", "どうして",
               "たしかに", "もし", "きっと", "まるで", "ちょうど", "もう", "まだ", "しまいに",
               "とき", "あいだ", "うち", "ため", "まま", "ほう", "とおり"}


def _name_like(subject: str) -> bool:
    """A caregiver question is only worth asking if the candidate answers look
    like real story entities, not parser debris ('取り扱う傾', 'けれど') or
    pronouns ('それ')."""
    if not subject or not (2 <= len(subject) <= 6):
        return False
    if _PARTICLE_CHARS & set(subject) or subject in _NON_ENTITY:
        return False
    # a relative-clause fragment ("同化しない間", "見事な牡鹿") carries verb/adjective
    # material -- the extractor's modifier stripper shortens it, or it embeds a
    # negation / verb ending mid-string
    from japanese_event_v1 import _strip_modifier
    if _strip_modifier(subject) != subject:
        return False
    if any(v in subject for v in ("ない", "しな", "する", "って", "たり", "ながら")):
        return False
    return not subject.startswith(("いきなり", "だんだん", "そのうち", "しばらく", "まもなく",
                                   "見事な", "りっぱな", "ふしぎな", "あわれな"))


def _fidelity_band(fidelity: float | None) -> int:
    """Map the automated retelling fidelity onto the 3-point human scale, so
    'human agreement' measures whether a person's coherence judgment tracks
    the metric."""
    if fidelity is None:
        return 2
    return 1 if fidelity >= 0.65 else 3 if fidelity < 0.3 else 2


def generate_questions(recent: list[dict], cycle: int, model=None) -> list[dict]:
    """recent: [{"title", "url", "events"}], most-recently-read first.

    The caregiver has NOT read these books, so every question is
    self-contained: it shows what Noise understood (its own retelling) and
    asks the reader to judge Noise's output, not to recall a story.

      * retelling_coherent -- is this Japanese coherent? (1 通る / 2 ときどき変 / 3 意味不明)
        graded against the automated fidelity band, so we learn whether a
        person's judgment tracks the metric.
      * protagonist_from_text -- reading only Noise's retelling, whose story is
        it?  Tests whether the retelling actually conveys the protagonist.
    """
    from japanese_retell_v1 import retell

    questions: list[dict] = []
    for book in recent:
        if len(questions) >= QUESTIONS_PER_BATCH:
            break
        events = [e for e in book.get("events", []) if e.get("verb")]
        if len(events) < 3:
            continue
        title = book.get("title") or "この本"
        url = book.get("url") or title
        retold = retell(events, max_sentences=6)
        if len(retold) < 20:
            continue
        excerpt = retold[:140]
        n = len(questions)

        # only real, recurring entities make a fair "whose story is it?" -- a
        # noun the parser produced once is debris, not a character
        counts = Counter(e.get("subject") for e in events if _name_like(e.get("subject") or ""))
        subjects = [s for s, c in counts.most_common() if c >= 2]
        if len(subjects) >= 2 and n % 2 == 0:
            protagonist = subjects[0]
            options = _distinct([protagonist] + subjects)[:3]
            options.sort(key=lambda o: hashlib.md5(f"{url}{o}".encode()).hexdigest())
            questions.append({
                "id": _qid(url, cycle, n), "cycle": cycle, "book": title, "url": url,
                "kind": "protagonist_from_text",
                "prompt": (f"Noiseが「{title}」を読んで、こう理解しました:\n"
                           f"    「{excerpt}」\n"
                           f"  この文章は だれの話に読めますか？　"
                           + "　".join(f"{i+1}) {o}" for i, o in enumerate(options))),
                "options": options,
                "answer_story": options.index(protagonist) + 1,
                "answer_model": (options.index(model.predict_protagonist(events[:2], options)) + 1)
                                if model is not None else None,
            })
            continue

        questions.append({
            "id": _qid(url, cycle, n), "cycle": cycle, "book": title, "url": url,
            "kind": "retelling_coherent",
            "prompt": (f"Noiseが「{title}」を読んで、こう再話しました:\n"
                       f"    「{excerpt}」\n"
                       f"  日本語として意味が通っていますか？　1) だいたい通る　2) ときどき変　3) ほとんど意味不明"),
            "options": ["1", "2", "3"],
            "answer_story": _fidelity_band(book.get("fidelity")),
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
