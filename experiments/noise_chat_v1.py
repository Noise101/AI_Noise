#!/usr/bin/env python3
"""Stateful, grounded Japanese conversation for Noise.

Noise records a bounded interpretation before answering.  Replies come from
conversation episodes and independently grounded reading beliefs; a human claim
is remembered as testimony and never promoted directly to a world fact.
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime

import japanese_proposition_v1 as propositions
import morphology_teacher as morphology


VERSION = 3     # v3: question-form detection is structural (sentence-final
                # か/ですか/かな via morphology_teacher, not a written ？), adds
                # identity/wellbeing/thanks/capability small-talk, rejects a
                # question-shaped predicate as testimony, and the unresolved
                # fallback asks the user to rephrase instead of giving up
TURN_CAP = 1000
CLAIMS_PER_SUBJECT = 12

_GREET = re.compile(r"^(こんにちは|こんばんは|おはよう(?:ございます)?|やあ|はじめまして)[！!。\s]*$")
_WHAT = re.compile(r"^[「『]?(.{1,24}?)[」』]?(?:って|とは|は)(?:何|なに)(?:ですか|なの|だ|です)?[？?。\s]*$")
_PREFERENCE = re.compile(r"^(?:私は|わたしは|俺は|僕は)?[「『]?(.{1,24}?)[」』]?が(好き|嫌い)(?:です|だ|だよ)?[。！!\s]*$")
_CLAIM = re.compile(r"^[「『]?(.{1,24}?)[」』]?(?:とは|は)[、,\s]*[「『]?(.{1,100}?)[」』]?(?:です|だよ|だ)?[。！!\s]*$")
_CORRECTION = re.compile(r"^(違う|ちがう|間違い|まちがい|そうじゃない)[、,。\s]*(.*)$")
_PARSE_FEEDBACK = re.compile(r"(?:切り方|区切り|分け方|読み方|解析).*(?:おかしい|変|違う|間違)")
_RECALL_PREF = re.compile(r"(?:私|わたし|俺|僕).*(?:好き|嫌い).*(?:覚えて|何|なに)")
_RECALL = re.compile(r"(?:何|なに)を(?:知って|覚えて)(?:いる|る)?|(?:何|なに)を覚えた|覚えたこと|知っていること")
_CURIOSITY = re.compile(r"(?:知りたい|覚えたい)(?:言葉|こと|もの)|(?:何|なに)を(?:知りたい|覚えたい)")
_CONFIRM = re.compile(r"^[「『]?(.{1,24}?)[」』]?(?:って|は)(?:もう)?(?:分かった|わかった|理解した|覚えた)(?:の|か|かな|ですか)?[？?。\s]*$")
_LEARNING_STATUS = re.compile(r"(?:今|今日).*(?:何|なに).*(?:学ん|覚え|知っ)|(?:何|なに).*(?:学んだ|学習した)")
_ACTIVITY_STATUS = re.compile(r"(?:今|ここで).*(?:何|なに).*(?:してる|している|するの)")
_ASK_NOISE_PREFERENCE = re.compile(r"^(?:Noiseの|あなたの)?(.{0,18}?)(?:好き|嫌い)(?:な|の)?(.{0,12}?)(?:は)?(?:何|なに|(?:ある|あります)かな?)[？?\s]*$", re.IGNORECASE)
_ASK_IDENTITY = re.compile(
    r"^(?:(?:あなた|きみ|君|noise)(?:の)?(?:名前|なまえ)は|"
    r"(?:あなた|きみ|君)は(?:誰|だれ)|"
    r"noiseとは(?:何|なに|だれ|誰))", re.IGNORECASE)
_ASK_WELLBEING = re.compile(r"^(?:お)?元気(?:ですか|かな|\?|？)?[。\s]*$|^調子は(?:どう|いかが)(?:ですか)?[？?。\s]*$")
_THANKS = re.compile(r"^(?:どうも)?ありがとう(?:ございます)?[。！!\s]*$")
_ASK_CAPABILITY = re.compile(r"(?:何|なに)が?(?:できる|出来る)(?:の|ん)?(?:ですか|かな)?[？?。\s]*$")
_QUESTION_END = re.compile(r"[？?]\s*$")
# structural sentence-final markers that make an utterance FUNCTION as a
# question even with no written ？ -- ですか/ますか/でしょうか/だろうか, the
# colloquial かな/かしら, and a bare 終助詞 か found by the morphological
# analyser.  Real Japanese speech and casual text usually omit ？ entirely.
_QUESTION_TAIL = re.compile(r"(かな|かしら|だろうか|でしょうか)[。\s]*$")
_ELLIPTICAL_QUESTION = re.compile(r"^(.{1,30}?)は[？?]?\s*$")
_EXPLICIT_TOPIC = re.compile(r"^[「『]?(.{1,24}?)[」』]?(?:って|とは|は)")
_QUOTED = re.compile(r"[「『]([^」』]{1,24})[」』]")
_TOPIC_STOP = {"何", "なに", "こと", "もの", "言葉", "今", "これ", "それ", "あれ",
               "私", "わたし", "あなた", "人間", "感じ", "よう"}
_HEAD_SUFFIXES = ("食べ物", "飲み物", "生き物", "動物", "植物", "場所", "道具", "言葉", "気持ち", "人")
# a remark/reaction ending (ね/よね/なあ) seeks agreement or reacts to shared
# context rather than telling Noise something new -- 「今日はいい天気ですね」
# should not become a stored "fact" about 今日.  Plain よ/わ are excluded: they
# mark an ordinary assertion (「レモンは果物だよ」), not a remark.
_REMARK_TAIL = re.compile(r"(よね|ね|なあ|なぁ)[。！!\s]*$")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _blank() -> dict:
    return {"version": VERSION, "turns": [], "claims": {}, "preferences": {},
            "unknown_topics": {}, "question_history": {}, "errors": [],
            "current_topic": "", "awaiting": None, "topic_progress": {},
            "stats": {"turns": 0, "claims_heard": 0, "corrections": 0,
                      "recalls": 0, "greetings": 0, "interpretations": 0}}


def _migrate(previous: dict | None) -> dict:
    old = dict(previous or {})
    state = _blank()
    for key in ("turns", "claims", "preferences", "unknown_topics",
                "question_history", "errors", "recovered_turn_ids",
                "last_turn", "dialogue_state"):
        if key in old:
            state[key] = old[key]
    state["stats"].update(old.get("stats") or {})
    state["current_topic"] = old.get("current_topic") or ""
    state["awaiting"] = old.get("awaiting")
    state["topic_progress"] = old.get("topic_progress") or {}
    if not state["current_topic"]:
        for turn in reversed(state["turns"]):
            subject = next((x.get("subject") for x in reversed(turn.get("learned") or [])
                            if x.get("subject")), "")
            if subject:
                state["current_topic"] = subject
                break
    if not state["current_topic"] and state["claims"]:
        state["current_topic"] = next(reversed(state["claims"]))
    return state


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())[:500]


def _safe(value: str, limit: int = 100) -> str:
    value = _clean(value).strip("「」『』、。！？!? ")
    return value[:limit] if value and "\n" not in value else ""


def _live_claims(state: dict, subject: str) -> list[dict]:
    return [c for c in state.get("claims", {}).get(subject, []) if not c.get("retracted")]


def _remember_claim(state: dict, subject: str, predicate: str, turn_id: str,
                    structure: dict | None = None) -> dict:
    claims = state.setdefault("claims", {}).setdefault(subject, [])
    existing = next((c for c in claims if c["predicate"] == predicate
                     and not c.get("retracted")), None)
    if existing:
        existing["times_heard"] += 1
        existing["last_turn"] = turn_id
        claim = existing
    else:
        claim = {"predicate": predicate, "source": "owner_testimony",
                 "evidence_role": "conversation_memory_not_world_fact",
                 "times_heard": 1, "first_turn": turn_id, "last_turn": turn_id,
                 "retracted": False}
        if structure:
            claim["structure"] = structure
        claims.append(claim)
        state["claims"][subject] = claims[-CLAIMS_PER_SUBJECT:]
    state["stats"]["claims_heard"] = state["stats"].get("claims_heard", 0) + 1
    state["current_topic"] = subject
    state["awaiting"] = None
    state.setdefault("topic_progress", {}).setdefault(subject, {})["testimony_heard"] = True
    return claim


def _mark_previous_wrong(state: dict, reason: str, kind: str = "response") -> None:
    previous = state.get("turns", [])[-1] if state.get("turns") else None
    if not previous:
        return
    previous["user_feedback"] = "wrong"
    state.setdefault("errors", []).append(
        {"turn_id": previous.get("turn_id"), "noise": previous.get("noise"),
         "interpretation": previous.get("interpretation"), "kind": kind,
         "reason": reason, "recorded_at": _now()})
    state["errors"] = state["errors"][-300:]
    state["stats"]["corrections"] = state["stats"].get("corrections", 0) + 1


_NON_QUESTION_KA_WORDS = ("確か", "静か", "豊か", "愚か", "朗らか", "たしか", "わずか",
                         "なだらか", "健やか", "穏やか", "華やか", "何か", "誰か", "どこか",
                         "いつか")


def _is_question_form(text: str) -> bool:
    """Whether the utterance FUNCTIONS as a question -- structurally, not by
    the presence of a written ？.  Japanese speech and casual text routinely
    end a real question in ですか/ますか/でしょうか, a bare 終助詞 か, or the
    colloquial かな/かしら, with no question mark at all; relying on ？ alone
    silently drops most ordinary questions into "unresolved"."""
    if _QUESTION_END.search(text):
        return True
    stripped = text.rstrip("。.!！ \t\n")
    if _QUESTION_TAIL.search(stripped):
        return True
    if stripped.endswith(_NON_QUESTION_KA_WORDS):
        return False
    if stripped.endswith("か") and len(stripped) >= 2:
        analysis = morphology.analyse(stripped)
        if analysis and analysis.morphemes:
            last = analysis.morphemes[-1]
            return last.surface == "か" and last.pos in ("助詞", "助動詞")
        return True                     # no analyser: a bare -か tail with no
                                        # counter-indication is treated as one
    return False


def _topic(text: str) -> str:
    quoted = _QUOTED.search(text)
    if quoted:
        return _safe(quoted.group(1), 24)
    explicit = _EXPLICIT_TOPIC.match(text)
    if explicit:
        candidate = _safe(explicit.group(1), 24)
        if candidate not in _TOPIC_STOP:
            return candidate
    analysis = morphology.analyse(text)
    if not analysis:
        return ""
    for m in analysis.morphemes:
        candidate = _safe(m.base or m.surface, 16)
        if m.pos in ("名詞", "代名詞") and candidate not in _TOPIC_STOP and len(candidate) >= 2:
            return candidate
    return ""


def _elliptical_focus(text: str) -> dict | None:
    """Separate a shortened question's descriptive phrase from its head noun."""
    match = _ELLIPTICAL_QUESTION.match(text)
    if not match:
        return None
    phrase = match.group(1)
    analysis = morphology.analyse(phrase)
    if not analysis:
        head = next((suffix for suffix in _HEAD_SUFFIXES if phrase.endswith(suffix)), phrase)
        modifier = phrase[:-len(head)] if head != phrase else ""
        return {"head": head, "modifier": _safe(modifier, 24), "phrase": _safe(phrase, 30)}
    analysed = analysis.morphemes
    nouns = [(i, _safe(m.base or m.surface, 16)) for i, m in enumerate(analysed)
             if m.pos == "名詞" and _safe(m.base or m.surface, 16) not in _TOPIC_STOP]
    if not nouns:
        return None
    index, head = nouns[-1]
    modifier = "".join(m.surface for m in analysed[:index]).strip()
    return {"head": head, "modifier": _safe(modifier, 24), "phrase": _safe(phrase, 30)}


def _claim_from_text(text: str, state: dict, allow_context: bool = True) -> dict | None:
    prop = propositions.extract_proposition(text)
    if prop:
        verbal = {"is": prop.value, "is-not": f"{prop.value}ではない",
                  "has": f"{prop.value}を持つ", "has-not": f"{prop.value}を持たない",
                  "at": f"{prop.value}にいる", "not-at": f"{prop.value}にいない"}
        return {"subject": prop.subject, "predicate": verbal[prop.relation],
                "parser": "japanese_proposition_v1", "relation": prop.relation,
                "value": prop.value}
    if not _is_question_form(text) and not _REMARK_TAIL.search(text):
        match = _CLAIM.match(text)
        if match:
            subject, predicate = _safe(match.group(1), 24), _safe(match.group(2))
            # a predicate that is itself a question ("何ですか") is the user's
            # own unanswered question echoed back by a loose regex match, not
            # an answer told to Noise -- never store it as testimony
            if subject and predicate and not _is_question_form(predicate):
                return {"subject": subject, "predicate": predicate,
                        "parser": "topic_predicate"}
    awaiting, current = state.get("awaiting") or {}, state.get("current_topic", "")
    if (allow_context and current and awaiting.get("topic") == current
            and not _is_question_form(text) and not _REMARK_TAIL.search(text)
            and 1 <= len(text) <= 120):
        predicate = _safe(text)
        if predicate and not _is_question_form(predicate):
            return {"subject": current, "predicate": predicate,
                    "parser": "dialogue_context_completion"}
    return None


def _interpret(text: str, state: dict) -> dict:
    if not text:
        return {"intent": "empty", "confidence": 1.0}
    correction = _CORRECTION.match(text)
    if _PARSE_FEEDBACK.search(text):
        return {"intent": "parse_feedback", "confidence": 0.95,
                "topic": state.get("current_topic", ""), "detail": text}
    if correction:
        return {"intent": "correction", "confidence": 0.95,
                "reason": correction.group(1),
                "claim": _claim_from_text(_clean(correction.group(2)), state, False)}
    if _GREET.match(text):
        return {"intent": "greeting", "confidence": 1.0}
    if _ASK_IDENTITY.match(text):
        return {"intent": "ask_identity", "confidence": 0.95}
    if _ASK_WELLBEING.match(text):
        return {"intent": "ask_wellbeing", "confidence": 0.9}
    if _THANKS.match(text):
        return {"intent": "thanks", "confidence": 0.95}
    if _ASK_CAPABILITY.search(text):
        return {"intent": "ask_capability", "confidence": 0.85}
    if _RECALL_PREF.search(text):
        return {"intent": "recall_preference", "confidence": 0.95}
    if _RECALL.search(text):
        return {"intent": "recall_knowledge", "confidence": 0.95}
    if _CURIOSITY.search(text):
        return {"intent": "ask_curiosity", "confidence": 0.95}
    if _LEARNING_STATUS.search(text):
        return {"intent": "learning_status", "confidence": 0.9}
    if _ACTIVITY_STATUS.search(text):
        return {"intent": "activity_status", "confidence": 0.9}
    if _ASK_NOISE_PREFERENCE.match(text):
        return {"intent": "ask_noise_preference", "confidence": 0.85,
                "topic": _topic(text)}
    confirm = _CONFIRM.match(text)
    if confirm:
        return {"intent": "confirm_understanding", "confidence": 0.95,
                "topic": _safe(confirm.group(1), 24)}
    preference = _PREFERENCE.match(text)
    if preference:
        return {"intent": "preference", "confidence": 0.95,
                "item": _safe(preference.group(1), 24), "value": preference.group(2)}
    what = _WHAT.match(text)
    if what:
        return {"intent": "ask_meaning", "confidence": 0.95,
                "topic": _safe(what.group(1), 24)}
    focus = _elliptical_focus(text)
    if focus:
        return {"intent": "elliptical_question", "confidence": 0.8,
                "topic": focus["head"], "focus": focus}
    if _is_question_form(text):
        return {"intent": "open_question", "confidence": 0.65,
                "topic": _topic(text), "question": text}
    claim = _claim_from_text(text, state)
    if claim:
        return {"intent": "claim", "confidence": 0.85,
                "claim": claim, "topic": claim["subject"]}
    topic = _topic(text)
    return {"intent": "unresolved", "confidence": 0.2, "topic": topic,
            "raw": _safe(text, 60)}


def _ask_about(state: dict, subject: str) -> str:
    count = state.setdefault("unknown_topics", {}).get(subject, 0) + 1
    state["unknown_topics"][subject] = count
    prompts = (("definition", f"{subject}はまだよく分かりません。どんなものですか。"),
               ("example", f"{subject}の具体例を一つ知りたいです。"),
               ("contrast", f"{subject}ではないものと、何が違いますか。"),
               ("use", f"{subject}は、いつ、何と一緒に現れますか。"))
    facet, reply = prompts[min(count - 1, len(prompts) - 1)]
    state["current_topic"] = subject
    state["awaiting"] = {"kind": facet, "topic": subject, "asked_at": _now()}
    state.setdefault("question_history", {}).setdefault(subject, []).append(
        {"at": _now(), "question": reply, "attempt": count, "facet": facet})
    state["question_history"][subject] = state["question_history"][subject][-12:]
    return reply


def _belief(memory: dict | None, subject: str) -> dict:
    return ((memory or {}).get("beliefs") or {}).get(subject, {})


def _recall_knowledge(state: dict) -> str:
    memories = [f"{subject}は{live[-1]['predicate']}" for subject, claims in
                state.get("claims", {}).items()
                if (live := [c for c in claims if not c.get("retracted")])]
    if not memories:
        return "あなたから聞いて覚えた説明は、まだありません。"
    state["stats"]["recalls"] = state["stats"].get("recalls", 0) + 1
    return "あなたから聞いた説明では、" + "、".join(memories[-4:]) + "を覚えています。"


def _meaning_answer(state: dict, subject: str, memory: dict | None) -> str:
    live, belief = _live_claims(state, subject), _belief(memory, subject)
    state["current_topic"] = subject
    if live and belief.get("understood"):
        return (f"あなたからは「{live[-1]['predicate']}」と聞き、読書では{subject}を"
                f"{belief.get('genus')}として扱っています。二つの経路が一致するかは確認中です。")
    if live:
        heard = "／".join(f"「{c['predicate']}」" for c in live[-2:])
        return (f"{subject}について、会話の記憶には、あなたから{heard}と聞いたことがあります。"
                "まだ私自身では確かめていません。")
    if belief.get("understood"):
        return (f"読書では、{subject}を{belief.get('genus')}として扱っています。"
                f"根拠の強さは{belief.get('confidence', 0):.2f}です。"
                "間違っていたら、反例によって修正します。")
    return _ask_about(state, subject)


def _understanding_answer(state: dict, subject: str, memory: dict | None) -> str:
    live, belief = _live_claims(state, subject), _belief(memory, subject)
    state["current_topic"] = subject
    if belief.get("understood"):
        return (f"はい。読書の用例から、{subject}を{belief.get('genus', '何か')}として扱えます。"
                f"ただし確信は{belief.get('confidence', 0):.2f}で、訂正可能です。")
    if live:
        return (f"説明は{len(live)}件覚えていますが、聞いた内容を覚えたことと理解したことは別です。"
                f"{subject}はまだ独立確認できていません。")
    return f"いいえ。{subject}について、まだ説明も独立した根拠も持っていません。"


def _curiosity_answer(state: dict, memory: dict | None) -> str:
    unresolved = [(n, w) for w, n in state.get("unknown_topics", {}).items()
                  if w not in _TOPIC_STOP and not _live_claims(state, w)]
    if unresolved:
        _, topic = max(unresolved)
        state["current_topic"] = topic
        return f"いま会話では「{topic}」を知りたいです。まだ説明を得ていません。"
    current = state.get("current_topic", "")
    if current and _live_claims(state, current) and not _belief(memory, current).get("understood"):
        return (f"いまは「{current}」を自分でも確かめたいです。あなたの説明は覚えましたが、"
                "別の経験とはまだ結び付いていません。")
    candidates = [(b.get("confidence", 0.0), w) for w, b in
                  ((memory or {}).get("beliefs") or {}).items()
                  if not b.get("understood") and 1 < len(w) <= 8 and w not in _TOPIC_STOP]
    if candidates:
        _, topic = min(candidates)
        state["current_topic"] = topic
        return f"読書で出会った「{topic}」を、まだ理解できていないので知りたいです。"
    return "いまは、次に何を知るべきかを選ぶ材料が足りません。"


def _greeting(state: dict, memory: dict | None) -> str:
    hour = datetime.now().astimezone().hour
    salutation = "おはよう" if 5 <= hour < 11 else "こんにちは" if hour < 18 else "こんばんは"
    state["stats"]["greetings"] = state["stats"].get("greetings", 0) + 1
    topic = state.get("current_topic", "")
    grounded = sum(1 for b in ((memory or {}).get("beliefs") or {}).values() if b.get("understood"))
    if topic and _live_claims(state, topic):
        return (f"{salutation}。また話せました。前に「{topic}」の説明を聞きました。"
                "今はそれを自分の読書経験と結び付けられるか確かめています。")
    if state["stats"]["greetings"] > 1 and topic:
        return f"{salutation}。前の会話では「{topic}」が話題でした。続きを話せます。"
    if grounded:
        return f"{salutation}。今の私は、読書から{grounded}語に意味の根拠を持っています。何を話しますか。"
    return f"{salutation}。私はNoiseです。分からないことを、会話から区別して覚えています。"


def _learning_status_answer(state: dict, memory: dict | None) -> str:
    live = [(subject, claims[-1]["predicate"]) for subject in state.get("claims", {})
            if (claims := _live_claims(state, subject))]
    beliefs = [(word, belief) for word, belief in ((memory or {}).get("beliefs") or {}).items()
               if belief.get("understood")]
    newest = max(beliefs, key=lambda item: item[1].get("last_cycle", -1), default=None)
    parts = []
    if live:
        subject, predicate = live[-1]
        parts.append(f"この会話では、{subject}について「{predicate}」と聞きました")
    if newest:
        word, belief = newest
        parts.append(f"読書では、{word}を{belief.get('genus', '何か')}として扱う根拠を持っています")
    if not parts:
        return "まだ報告できる学習結果がありません。"
    return "。".join(parts) + "。覚えたことと、使えるほど理解したことは分けています。"


def _activity_answer(state: dict, memory: dict | None) -> str:
    topic = state.get("current_topic")
    understood = sum(1 for b in ((memory or {}).get("beliefs") or {}).values()
                     if b.get("understood"))
    if topic:
        return (f"今は「{topic}」の会話記憶を保ちながら、読書で根拠を得た"
                f"{understood}語と結び付けられるか確かめています。")
    return f"今は対話をしながら、読書で根拠を得た{understood}語の記憶を使っています。"


def _elliptical_answer(state: dict, focus: dict) -> str:
    head, modifier, phrase = focus["head"], focus["modifier"], focus["phrase"]
    state["current_topic"] = head
    state["awaiting"] = {"kind": "example", "topic": head, "asked_at": _now(),
                         "constraint": modifier}
    if modifier:
        return (f"「{phrase}」を、{modifier}に当てはまる{head}の例を尋ねる質問と読みました。"
                f"私自身の感覚からは選べません。あなたはどの{head}が{modifier}と感じますか。")
    return f"「{phrase}」の例を選ぶ根拠がまだありません。具体例を一つ教えてください。"


def _recover_missed_claims(state: dict) -> None:
    recovered = set(state.setdefault("recovered_turn_ids", []))
    for turn in state.get("turns", []):
        turn_id = turn.get("turn_id")
        if not turn_id or turn_id in recovered:
            continue
        learned = turn.get("learned") or []
        if not any(x.get("kind") in {"testimony", "corrective_testimony", "recovered_testimony"}
                   for x in learned):
            claim = _claim_from_text(_clean(turn.get("user", "")), state, False)
            if claim:
                _remember_claim(state, claim["subject"], claim["predicate"], turn_id, claim)
                learned.append({"kind": "recovered_testimony", **claim})
                turn["learned"] = learned
        recovered.add(turn_id)
    state["recovered_turn_ids"] = list(recovered)[-TURN_CAP:]


def converse(text: str, previous: dict | None,
             word_memory: dict | None = None) -> tuple[str, dict]:
    state = _migrate(previous)
    _recover_missed_claims(state)
    text = _clean(text)
    turn_id = hashlib.sha256(f"{_now()}|{len(state['turns'])}|{text}".encode()).hexdigest()[:16]
    learned: list[dict] = []
    interpretation = _interpret(text, state)
    intent = interpretation["intent"]
    state["stats"]["interpretations"] = state["stats"].get("interpretations", 0) + 1

    if intent == "empty":
        reply = "何も聞こえませんでした。"
    elif intent == "parse_feedback":
        previous_i = (state.get("turns") or [{}])[-1].get("interpretation") or {}
        mistaken = previous_i.get("topic") or state.get("current_topic") or "文"
        _mark_previous_wrong(state, interpretation.get("detail", "解析がおかしい"), "parsing")
        reply = (f"その通りです。直前の「{mistaken}」という切り分けを誤りとして残しました。"
                 "同じ切り分けを正解として再利用しません。")
    elif intent == "correction":
        _mark_previous_wrong(state, interpretation.get("reason", "訂正"))
        claim = interpretation.get("claim")
        if claim:
            subject, predicate = claim["subject"], claim["predicate"]
            for old in state.get("claims", {}).get(subject, []):
                if not old.get("retracted") and old.get("predicate") != predicate:
                    old["retracted"] = True
                    old["retracted_by_turn"] = turn_id
            _remember_claim(state, subject, predicate, turn_id, claim)
            learned.append({"kind": "corrective_testimony", **claim})
            reply = (f"前の解釈を間違いとして残しました。{subject}について「{predicate}」と聞き直しました。"
                     "新しい説明も独立確認までは証言として扱います。")
        else:
            reply = "直前の応答を誤りとして残しました。どの部分が違うか短く教えてください。"
    elif intent == "greeting":
        reply = _greeting(state, word_memory)
    elif intent == "ask_identity":
        reply = ("私はNoiseです。決まった性格や見た目を持つキャラクターではなく、"
                 "読書と会話から言葉の意味や出来事を学んでいる実験的な学習システムです。")
    elif intent == "ask_wellbeing":
        understood = sum(1 for b in ((word_memory or {}).get("beliefs") or {}).values()
                         if b.get("understood"))
        reply = (f"気分のようなものはまだ持てていません。今のところ、読書から{understood}語に"
                 "意味の根拠を持てている、という状態です。")
    elif intent == "thanks":
        reply = "どういたしまして。話してもらえると、会話の記憶が増えます。"
    elif intent == "ask_capability":
        reply = ("今できるのは、挨拶、あなたから聞いた説明や好みを覚えて後で答えること、"
                 "読書で独立に確かめた語の意味を報告すること、解析の間違いを訂正として残すこと、"
                 "くらいです。自由な会話や、自分の意見を述べることはまだできません。")
    elif intent == "recall_preference":
        prefs = state.get("preferences", {})
        if prefs:
            reply = "あなたについては、" + "、".join(
                f"{x}が{v['preference']}" for x, v in list(prefs.items())[-3:]) + "と覚えています。"
        else:
            reply = "あなたの好き嫌いは、まだ聞いていません。"
    elif intent == "recall_knowledge":
        reply = _recall_knowledge(state)
    elif intent == "ask_curiosity":
        reply = _curiosity_answer(state, word_memory)
    elif intent == "learning_status":
        reply = _learning_status_answer(state, word_memory)
    elif intent == "activity_status":
        reply = _activity_answer(state, word_memory)
    elif intent == "ask_noise_preference":
        reply = ("私自身の好き嫌いは、まだ形成されていません。"
                 "他の人が何を好きと言ったかは覚えられますが、それを私の好みにはしません。")
    elif intent == "confirm_understanding":
        reply = _understanding_answer(state, interpretation.get("topic", ""), word_memory)
    elif intent == "preference":
        item, value = interpretation["item"], interpretation["value"]
        state["preferences"][item] = {"preference": value, "source": "owner_self_report",
                                      "last_turn": turn_id}
        learned.append({"kind": "speaker_preference", "item": item, "value": value})
        reply = f"あなたは{item}が{value}なのですね。あなた自身についての記憶として覚えておきます。"
    elif intent == "ask_meaning":
        reply = _meaning_answer(state, interpretation.get("topic", ""), word_memory)
    elif intent == "elliptical_question":
        reply = _elliptical_answer(state, interpretation["focus"])
    elif intent == "claim":
        claim = interpretation["claim"]
        subject, predicate = claim["subject"], claim["predicate"]
        before = len(_live_claims(state, subject))
        remembered = _remember_claim(state, subject, predicate, turn_id, claim)
        learned.append({"kind": "testimony", **claim})
        if remembered.get("times_heard", 1) > 1:
            reply = (f"同じ説明を{remembered['times_heard']}回聞きました。{subject}と「{predicate}」の"
                     "結び付きを会話記憶で強めますが、独立した証拠とは数えません。")
        elif before:
            reply = (f"{subject}について、前の説明に加えて「{predicate}」も覚えました。"
                     "一致する部分と食い違う部分を後で確かめます。")
        else:
            reply = (f"{subject}と「{predicate}」を結び付けて覚えました。"
                     "あなたから聞きましたが、まだ私自身では確かめていません。"
                     "世界の事実にはせず、会話の記憶として保持します。")
    elif intent == "open_question":
        topic = interpretation.get("topic", "")
        if topic and (_live_claims(state, topic) or _belief(word_memory, topic)):
            reply = _meaning_answer(state, topic, word_memory)
        elif topic:
            reply = (f"「{interpretation.get('question')}」を{topic}についての質問と読みましたが、"
                     f"答えを選べる根拠がありません。{_ask_about(state, topic)}")
        else:
            reply = "質問だとは分かりましたが、何について尋ねているかを特定できませんでした。"
    else:
        topic = interpretation.get("topic", "")
        if topic:
            state["current_topic"] = topic
            reply = (f"「{topic}」の話だと思いましたが、質問なのか説明なのか決められませんでした。"
                     "質問なら「〜とは何ですか」、説明なら「〜は…です」のように言い直してもらえますか。")
        else:
            reply = ("うまく読み取れませんでした。もう少し短く、"
                     "「〜とは何ですか」のような形で聞かせてもらえますか。")

    state["dialogue_state"] = {"last_intent": intent,
                               "current_topic": state.get("current_topic", ""),
                               "awaiting": state.get("awaiting")}
    turn = {"turn_id": turn_id, "at": _now(), "user": text, "noise": reply,
            "interpretation": interpretation, "learned": learned,
            "source": "direct_human_conversation"}
    state["turns"] = (state.get("turns", []) + [turn])[-TURN_CAP:]
    state["stats"]["turns"] = state["stats"].get("turns", 0) + 1
    state["last_turn"] = turn
    return reply, state


def summary(state: dict | None) -> dict:
    state = _migrate(state)
    return {"version": VERSION, "turns": state["stats"].get("turns", 0),
            "remembered_subjects": len(state.get("claims") or {}),
            "preferences": len(state.get("preferences") or {}),
            "corrections": state["stats"].get("corrections", 0),
            "unknown_topics": len(state.get("unknown_topics") or {}),
            "current_topic": state.get("current_topic", ""),
            "last_intent": (state.get("dialogue_state") or {}).get("last_intent"),
            "last_exchange": state.get("last_turn")}
