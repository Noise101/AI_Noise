#!/usr/bin/env python3
"""Minimal meaningful conversation backed by explicit episodic memory.

Noise is not ready for unconstrained generation.  This interface therefore
prefers a short grounded reply, an honest uncertainty, or a question.  Human
statements are remembered as *testimony about the conversation*, never copied
into world knowledge.  Repetition, correction and later recall are learned
behaviours without pretending that a speaker's claim is objective truth.
"""

from __future__ import annotations

import hashlib
import re
import time

VERSION = 1
TURN_CAP = 1000
CLAIMS_PER_SUBJECT = 12

_GREET = re.compile(r"^(こんにちは|こんばんは|おはよう|やあ|はじめまして)[！!。\s]*$")
_WHAT = re.compile(r"^[「『]?(.{1,16}?)[」』]?(?:って|とは|は)(?:何|なに)(?:ですか|なの|だ|です)?[？?。\s]*$")
_PREFERENCE = re.compile(r"^(?:私は|わたしは|俺は|僕は)?[「『]?(.{1,16}?)[」』]?が(好き|嫌い)(?:です|だ|だよ)?[。！!\s]*$")
_CLAIM = re.compile(
    r"^[「『]?(.{1,16}?)[」』]?(?:とは|は)[、,\s]*"
    r"[「『]?(.{1,80}?)[」』]?(?:です|だよ|だ)?[。！!\s]*$")
_CORRECTION = re.compile(r"^(違う|ちがう|間違い|まちがい|そうじゃない)[、,。\s]*(.*)$")
_RECALL_PREF = re.compile(r"(?:私|わたし|俺|僕).*(?:好き|嫌い).*(?:覚えて|何|なに)")
_RECALL_KNOWLEDGE = re.compile(r"(?:何|なに)を(?:知って|覚えて)(?:いる|る)?")
_TOPIC = re.compile(r"[「『]([^」』]{1,16})[」』]|([一-鿿々]{2,8})|([ァ-ヿー]{2,12})")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _blank() -> dict:
    return {"version": VERSION, "turns": [], "claims": {}, "preferences": {},
            "unknown_topics": {}, "question_history": {}, "errors": [], "stats": {"turns": 0,
            "claims_heard": 0, "corrections": 0, "recalls": 0}}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())[:500]


def _safe_phrase(value: str, limit: int) -> str:
    value = _clean(value).strip("「」『』、。！？!? ")
    return value[:limit] if value and "\n" not in value else ""


def _remember_claim(state: dict, subject: str, predicate: str, turn_id: str) -> dict:
    claims = state.setdefault("claims", {}).setdefault(subject, [])
    existing = next((c for c in claims if c["predicate"] == predicate and not c.get("retracted")), None)
    if existing:
        existing["times_heard"] += 1
        existing["last_turn"] = turn_id
        claim = existing
    else:
        claim = {"predicate": predicate, "source": "owner_testimony",
                 "evidence_role": "conversation_memory_not_world_fact",
                 "times_heard": 1, "first_turn": turn_id, "last_turn": turn_id,
                 "retracted": False}
        claims.append(claim)
        state["claims"][subject] = claims[-CLAIMS_PER_SUBJECT:]
    state["stats"]["claims_heard"] += 1
    return claim


def _latest_claim(state: dict, subject: str) -> dict | None:
    live = [c for c in state.get("claims", {}).get(subject, []) if not c.get("retracted")]
    return live[-1] if live else None


def _mark_previous_wrong(state: dict, reason: str) -> None:
    turns = state.get("turns", [])
    previous = turns[-1] if turns else None
    if not previous:
        return
    previous["user_feedback"] = "wrong"
    err = {"turn_id": previous["turn_id"], "noise": previous["noise"],
           "reason": reason, "recorded_at": _now()}
    state.setdefault("errors", []).append(err)
    state["errors"] = state["errors"][-300:]
    state["stats"]["corrections"] += 1


def _topic(text: str) -> str:
    m = _TOPIC.search(text)
    return next((g for g in m.groups() if g), "") if m else ""


def _ask_about(state: dict, subject: str) -> str:
    """Ask for missing evidence without repeating the same prompt forever."""
    count = state["unknown_topics"].get(subject, 0) + 1
    state["unknown_topics"][subject] = count
    prompts = (
        f"{subject}は、まだよく分かりません。どんなものですか。",
        f"{subject}について前にも尋ねました。具体例を一つ教えてください。",
        f"{subject}をまだ理解できていません。何に似ていて、いつ使いますか。",
    )
    reply = prompts[min(count - 1, len(prompts) - 1)]
    state.setdefault("question_history", {}).setdefault(subject, []).append(
        {"at": _now(), "question": reply, "attempt": count})
    state["question_history"][subject] = state["question_history"][subject][-12:]
    return reply


def _recall_knowledge(state: dict) -> str:
    memories = []
    for subject, claims in state.get("claims", {}).items():
        live = [claim for claim in claims if not claim.get("retracted")]
        if live:
            memories.append(f"{subject}は{live[-1]['predicate']}")
    if not memories:
        return "あなたから聞いて覚えた説明は、まだありません。"
    state["stats"]["recalls"] += 1
    return "あなたから聞いたことでは、" + "、".join(memories[-4:]) + "を覚えています。"


def _recover_missed_claims(state: dict) -> None:
    """Recover definitions missed by the older, copula-only parser once."""
    recovered = set(state.setdefault("recovered_turn_ids", []))
    for turn in state.get("turns", []):
        turn_id = turn.get("turn_id")
        if not turn_id or turn_id in recovered:
            continue
        learned = turn.get("learned") or []
        if any(item.get("kind") in {"testimony", "corrective_testimony"} for item in learned):
            recovered.add(turn_id)
            continue
        utterance = _clean(turn.get("user", ""))
        match = None if utterance.endswith(("?", "？")) else _CLAIM.match(utterance)
        if match:
            subject = _safe_phrase(match.group(1), 16)
            predicate = _safe_phrase(match.group(2), 80)
            if subject and predicate:
                _remember_claim(state, subject, predicate, turn_id)
                learned.append({"kind": "recovered_testimony", "subject": subject,
                                "predicate": predicate})
                turn["learned"] = learned
        recovered.add(turn_id)
    state["recovered_turn_ids"] = list(recovered)[-TURN_CAP:]


def converse(text: str, previous: dict | None, word_memory: dict | None = None) -> tuple[str, dict]:
    state = dict(previous or {})
    if state.get("version") != VERSION:
        state = _blank()
    for k, v in _blank().items():
        state.setdefault(k, v)
    _recover_missed_claims(state)
    text = _clean(text)
    turn_id = hashlib.sha256(f"{_now()}|{len(state['turns'])}|{text}".encode()).hexdigest()[:16]
    learned = []

    correction = _CORRECTION.match(text)
    if correction:
        _mark_previous_wrong(state, correction.group(1))
        remainder = _clean(correction.group(2))
        claim_match = _CLAIM.match(remainder)
        if claim_match:
            subject, predicate = (_safe_phrase(claim_match.group(1), 16),
                                  _safe_phrase(claim_match.group(2), 32))
            if subject and predicate:
                for old in state.get("claims", {}).get(subject, []):
                    if not old.get("retracted") and old.get("predicate") != predicate:
                        old["retracted"] = True
                        old["retracted_by_turn"] = turn_id
                        old["retraction_reason"] = "explicit_owner_correction"
                _remember_claim(state, subject, predicate, turn_id)
                learned.append({"kind": "corrective_testimony", "subject": subject,
                                "predicate": predicate})
                reply = f"前の言い方を間違いとして残しました。{subject}は{predicate}だと聞き直しました。"
            else:
                reply = "前の言い方を間違いとして残しました。正しい言い方を教えてください。"
        else:
            reply = "前の言い方を間違いとして残しました。正しい言い方を教えてください。"
    elif not text:
        reply = "まだ何も聞こえません。短い文で話してください。"
    elif _GREET.match(text):
        reply = "こんにちは。私はNoiseです。まだ短い会話を練習しています。"
    elif _RECALL_PREF.search(text):
        prefs = state.get("preferences", {})
        if prefs:
            bits = [f"{x}が{v['preference']}" for x, v in list(prefs.items())[-3:]]
            reply = "あなたは" + "、".join(bits) + "と聞いたことを覚えています。"
            state["stats"]["recalls"] += 1
        else:
            reply = "あなたの好き嫌いは、まだ聞いていません。"
    elif _RECALL_KNOWLEDGE.search(text):
        reply = _recall_knowledge(state)
    else:
        pref = _PREFERENCE.match(text)
        what = _WHAT.match(text)
        claim_match = _CLAIM.match(text) if not text.endswith(("?", "？")) else None
        if pref:
            item, value = _safe_phrase(pref.group(1), 16), pref.group(2)
            state["preferences"][item] = {"preference": value, "source": "owner_self_report",
                                          "last_turn": turn_id}
            learned.append({"kind": "speaker_preference", "item": item, "value": value})
            reply = f"あなたは{item}が{value}なのですね。覚えておきます。"
        elif what:
            subject = _safe_phrase(what.group(1), 16)
            remembered = _latest_claim(state, subject)
            belief = ((word_memory or {}).get("beliefs") or {}).get(subject, {})
            if remembered:
                reply = (f"あなたから、{subject}は{remembered['predicate']}だと聞きました。"
                         "これは会話の記憶で、まだ私自身では確かめていません。")
                state["stats"]["recalls"] += 1
            elif belief.get("understood") and belief.get("genus"):
                reply = (f"読書では、{subject}を{belief['genus']}として扱っています。"
                         "間違っていたら教えてください。")
            else:
                reply = _ask_about(state, subject)
        elif claim_match:
            subject, predicate = (_safe_phrase(claim_match.group(1), 16),
                                  _safe_phrase(claim_match.group(2), 80))
            if subject and predicate:
                rec = _remember_claim(state, subject, predicate, turn_id)
                learned.append({"kind": "testimony", "subject": subject, "predicate": predicate})
                repeated = "もう一度" if rec["times_heard"] > 1 else ""
                reply = (f"{subject}は{predicate}だと{repeated}聞きました。"
                         "まだ私自身では確かめていません。")
            else:
                reply = "その文をうまく分けられませんでした。もっと短く教えてください。"
        else:
            topic = _topic(text)
            if topic:
                reply = _ask_about(state, topic)
            else:
                reply = "まだその文を理解できません。主語と内容を短く教えてください。"

    turn = {"turn_id": turn_id, "at": _now(), "user": text, "noise": reply,
            "learned": learned, "source": "direct_human_conversation"}
    state["turns"] = (state.get("turns", []) + [turn])[-TURN_CAP:]
    state["stats"]["turns"] += 1
    state["last_turn"] = turn
    return reply, state


def summary(state: dict | None) -> dict:
    state = state or {}
    return {"turns": (state.get("stats") or {}).get("turns", 0),
            "remembered_subjects": len(state.get("claims") or {}),
            "preferences": len(state.get("preferences") or {}),
            "corrections": (state.get("stats") or {}).get("corrections", 0),
            "unknown_topics": len(state.get("unknown_topics") or {}),
            "last_exchange": state.get("last_turn")}
