#!/usr/bin/env python3
"""Stateful, grounded Japanese conversation for Noise.

Noise records a bounded interpretation before answering.  Replies come from
conversation episodes and independently grounded reading beliefs; a human claim
is remembered as testimony and never promoted directly to a world fact.

A local model may re-express an already-fully-decided reply in more natural
Japanese (`PhrasingModel` / `_phrase_naturally`) -- it is a presentation layer,
never a source of content: every content word in the template reply must
survive verbatim in the rephrase or the template is used unchanged.  This is
the same evidence-score-0 boundary as the rest of the project's local-model
uses (ARCHITECTURE.md "Optional local-model boundary"): the LLM never decides
WHAT Noise says, only offers one way of saying it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from datetime import datetime

import conversation_embedding_v1 as association
import japanese_proposition_v1 as propositions
import morphology_teacher as morphology


VERSION = 6     # v6: every remaining _interpret intent (correction, parse
                # feedback, recall, curiosity, learning/activity status,
                # Noise's-own-preference, confirm-understanding, preference,
                # "Xとは何", the generic claim split) converted from a literal
                # surface regex to lemma/structure-first matching, regex kept
                # only as the no-analyser fallback; _topic finds は/って/とは
                # as an actual 助詞 token, not a character search (which
                # matched って embedded inside an unrelated word's て-form);
                # recall_knowledge answers the asked subject, not every
                # subject Noise holds a claim about; bare pronouns (それ/こ
                # れ/あれ/私/あなた) rejected as a claim's subject key; greeting
                # echoes the user's own time-of-day, not the wall clock
                # v5: conversation_embedding_v1 gives a bounded, self-trained
                # associative recall memory ("we talked about X before, is
                # this related?") over topics/claims already held -- never a
                # source of content, see _recall_prompt
                # v4: topic_stack gives multi-topic memory (follow_up,
                # go_back_topic) instead of a single current_topic string;
                # PhrasingModel/_phrase_naturally lets a local model re-express
                # an already-decided reply more naturally, content-verified
                # against the template it may replace (never a content source)
                # v3: question-form detection is structural (sentence-final
                # か/ですか/かな via morphology_teacher, not a written ？), adds
                # identity/wellbeing/thanks/capability small-talk, rejects a
                # question-shaped predicate as testimony, and the unresolved
                # fallback asks the user to rephrase instead of giving up
TURN_CAP = 1000
CLAIMS_PER_SUBJECT = 12
TOPIC_STACK_CAP = 8

_GREET = re.compile(r"^(こんにち[はわ]|こんばん[はわ]|おはよう(?:ございます)?|やあ|"
                    r"はじめまして)[！!。\s]*$")
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
# multi-topic conversational memory: a follow-up that continues the CURRENT
# topic ("もっと教えて", "それについては？", "他には？") vs. one that explicitly
# returns to an EARLIER topic ("さっきの話に戻って", "犬の話に戻ろう") -- both
# need `topic_stack`, not just the single `current_topic` string, to mean
# anything.
_FOLLOW_UP = re.compile(
    r"^(?:それ(?:について)?|そのこと)?(?:もっと|他には|ほかには|さらに|続けて)"
    r"(?:教えて|話して|ある)?[。！!？?\s]*$|"
    r"^詳しく(?:教えて)?[。！!？?\s]*$|^それは[？?]?[。\s]*$|^それも[。！!？?\s]*$")
_GO_BACK_NAMED = re.compile(r"^(.{1,20}?)の話に戻(?:って|ろう)[。！!\s]*$")
_GO_BACK_GENERIC = re.compile(
    r"^(?:さっき|前|元|最初)の話(?:に戻(?:って|ろう)|は)?[？?。\s]*$")

# the reply is not parsed by the phrasing check, only compared -- approximate
# its content words as kanji/katakana runs, same technique as
# japanese_dialogue_v1._content_terms (particles/okurigana are hiragana and
# drop out, which is exactly what should be free to change in a rephrase)
_KANJI_RUN = re.compile(r"[一-鿿々]+")
_KATA_RUN = re.compile(r"[゠-ヿー]{2,}")


def _content_terms(text: str) -> set[str]:
    return set(_KANJI_RUN.findall(text)) | set(_KATA_RUN.findall(text))


class PhrasingModel:
    """Local-model presentation layer for an already-decided reply.  Same
    boundary as `japanese_dialogue_v1.JapanesePartner`: evidence score 0,
    never a source of content.  `rephrase` returns None on any failure so the
    caller always has a safe original template to fall back to."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model or os.environ.get("AI_NOISE_LOCAL_MODEL", "qwen3.8:27b")

    def available(self) -> bool:
        if os.environ.get("AI_NOISE_SKIP_LOCAL_LLM"):
            return False
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2) as r:
                return any(m.get("name") == self.model for m in json.load(r).get("models", []))
        except Exception:
            return False

    def rephrase(self, template: str) -> str | None:
        prompt = ("次の日本語の文を、内容と事実を一切変えずに、より自然な話し言葉に"
                  "言い換えてください。新しい情報や意見を足さないでください。数字や"
                  "固有名詞はそのまま残してください。英語は使わないでください。\n"
                  f"元の文: {template[:400]}")
        schema = {"type": "object", "properties": {"reply": {"type": "string"}},
                  "required": ["reply"]}
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                              "think": False, "format": schema,
                              "options": {"temperature": 0.3, "num_predict": 200}}).encode()
        req = urllib.request.Request(f"{self.base_url}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=None) as r:
                parsed = json.loads(json.load(r).get("response", "{}"))
            return str(parsed.get("reply", "")).strip()[:600] or None
        except Exception:
            return None


def _phrase_naturally(template: str, phraser: "PhrasingModel | None" = None) -> str:
    """Re-express `template` more naturally IF a verified rephrase is
    available; otherwise return it unchanged.  The rephrase is accepted only
    if every content word (kanji/katakana run) of the template survives in it
    verbatim and its length is not wildly different -- a cheap, mechanical
    content-fidelity check, not a semantic one, but sufficient to catch the
    failure modes that matter: dropped facts, invented facts, or a bad/empty
    generation. AI_NOISE_CHAT_PHRASING=0 disables this layer entirely."""
    if os.environ.get("AI_NOISE_CHAT_PHRASING") == "0" or not template:
        return template
    phraser = phraser if phraser is not None else PhrasingModel()
    if not phraser.available():
        return template
    required = _content_terms(template)
    if not required:
        return template
    rephrased = phraser.rephrase(template)
    if not rephrased or not required <= _content_terms(rephrased):
        return template
    if not (0.4 * len(template) <= len(rephrased) <= 3 * len(template)):
        return template
    return rephrased


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _blank() -> dict:
    return {"version": VERSION, "turns": [], "claims": {}, "preferences": {},
            "unknown_topics": {}, "question_history": {}, "errors": [],
            "current_topic": "", "awaiting": None, "topic_progress": {},
            "topic_stack": [],
            "stats": {"turns": 0, "claims_heard": 0, "corrections": 0,
                      "recalls": 0, "greetings": 0, "interpretations": 0}}


def _touch_topic(state: dict, topic: str) -> None:
    """Make `topic` current and move it to the top of the recency stack --
    the structural memory that lets a later turn say 「さっきの話に戻って」 or
    「それについてもっと」 and mean something, instead of only ever knowing the
    single most recent topic."""
    if not topic:
        return
    state["current_topic"] = topic
    stack = [t for t in state.setdefault("topic_stack", []) if t != topic]
    stack.append(topic)
    state["topic_stack"] = stack[-TOPIC_STACK_CAP:]


def _migrate(previous: dict | None) -> dict:
    old = dict(previous or {})
    state = _blank()
    for key in ("turns", "claims", "preferences", "unknown_topics",
                "question_history", "errors", "recovered_turn_ids",
                "last_turn", "dialogue_state", "embedding_memory"):
        if key in old:
            state[key] = old[key]
    state["stats"].update(old.get("stats") or {})
    state["current_topic"] = old.get("current_topic") or ""
    state["awaiting"] = old.get("awaiting")
    state["topic_progress"] = old.get("topic_progress") or {}
    state["topic_stack"] = [t for t in (old.get("topic_stack") or []) if t][-TOPIC_STACK_CAP:]
    if not state["current_topic"]:
        for turn in reversed(state["turns"]):
            subject = next((x.get("subject") for x in reversed(turn.get("learned") or [])
                            if x.get("subject")), "")
            if subject:
                state["current_topic"] = subject
                break
    if not state["current_topic"] and state["claims"]:
        state["current_topic"] = next(reversed(state["claims"]))
    if not state["topic_stack"] and state["current_topic"]:
        state["topic_stack"] = [state["current_topic"]]
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
    _touch_topic(state, subject)
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
    end a real question in ですか/ますか/でしょうか, a bare 終助詞 か or の, or
    the colloquial かな/かしら, with no question mark at all; relying on ？
    alone silently drops most ordinary questions into "unresolved"."""
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
    if stripped.endswith("の") and len(stripped) >= 2:
        # 終助詞 の ("何ができるの") is a casual question; 連体化 の ("私のもの
        # の", nominalising/possessive) is not -- only the analyser tells
        # them apart, so with none installed a bare の is left unresolved
        # rather than risk reading every attributive の as a question
        analysis = morphology.analyse(stripped)
        if analysis and analysis.morphemes:
            last = analysis.morphemes[-1]
            return (last.surface == "の" and last.pos == "助詞"
                    and last.pos_detail == "終助詞")
        return False
    return False


# --- structural small-talk detection ----------------------------------------
# Matching a fixed surface string ("ありがとうございます") is exactly why
# "こんばんわ" (spelling) and "何が出来るの" (a conjugation regex missed) kept
# failing live -- every new ending/politeness level/spelling is its own patch
# forever.  These check morphology's LEMMA (dictionary form, invariant across
# conjugation, honorific prefixes, and politeness level) and structural shape
# instead, with the original literal regex kept as the fallback when no
# analyser is installed (AI_NOISE_NO_MORPHOLOGY=1) -- invariant 1 holds.
_GREETING_LEMMAS = {"こんにちは", "こんにちわ", "こんばんは", "こんばんわ",
                    "おはよう", "やあ", "はじめまして"}
_THANKS_LEMMAS = {"ありがとう", "有難う"}
_WELLBEING_LEMMAS = {"元気", "調子"}
_CAPABILITY_LEMMAS = {"できる", "出来る"}
_IDENTITY_PRONOUNS = {"あなた", "きみ", "君", "noise"}
_IDENTITY_NOUNS = {"名前", "なまえ"}
_TRAILING_POS = ("助動詞", "記号", "助詞")


def _lemmas_of(morphemes: list, pos: "tuple[str, ...]") -> set[str]:
    return {(m.base or m.surface) for m in morphemes if m.pos in pos}


def _leading_interjection(morphemes: list) -> "tuple[object, int] | None":
    """The first 感動詞, skipping a leading intensifying adverb ("どうも
    ありがとう", "本当にありがとう") -- returns (morpheme, index) or None."""
    for i, m in enumerate(morphemes):
        if m.pos == "副詞":
            continue
        return (m, i) if m.pos == "感動詞" else None
    return None


def _is_greeting(text: str) -> bool:
    a = morphology.analyse(text.rstrip("。.!！ \t\n"))
    if not a or not a.morphemes:
        return bool(_GREET.match(text))
    hit = _leading_interjection(a.morphemes)
    if not hit or (hit[0].base or hit[0].surface) not in _GREETING_LEMMAS:
        return False
    return all(m.pos in _TRAILING_POS for m in a.morphemes[hit[1] + 1:])   # +ございます等


def _is_thanks(text: str) -> bool:
    a = morphology.analyse(text.rstrip("。.!！ \t\n"))
    if not a or not a.morphemes:
        return bool(_THANKS.match(text))
    hit = _leading_interjection(a.morphemes)
    if not hit or (hit[0].base or hit[0].surface) not in _THANKS_LEMMAS:
        return False
    return all(m.pos in _TRAILING_POS for m in a.morphemes[hit[1] + 1:])   # +ございました等


def _is_wellbeing_question(text: str) -> bool:
    a = morphology.analyse(text.rstrip("。.!！？? \t\n"))
    if not a or not a.morphemes:
        return bool(_ASK_WELLBEING.match(text))
    lemmas = _lemmas_of(a.morphemes, ("名詞",))
    if not (lemmas & _WELLBEING_LEMMAS):
        return False
    # 元気 alone (as a claim -- "元気な子") is not this; it must stand as the
    # sentence's own topic/predicate, not a modifier before another noun
    return len(a.morphemes) <= 6 and _is_question_form(text)


def _is_capability_question(text: str) -> bool:
    a = morphology.analyse(text.rstrip("。.!！？? \t\n"))
    if not a or not a.morphemes:
        return bool(_ASK_CAPABILITY.search(text))
    lemmas = _lemmas_of(a.morphemes, ("動詞",))
    return bool(lemmas & _CAPABILITY_LEMMAS) and _is_question_form(text)


def _is_identity_question(text: str) -> bool:
    a = morphology.analyse(text.rstrip("。.!！？? \t\n"))
    if not a or not a.morphemes:
        return bool(_ASK_IDENTITY.match(text))
    ms = a.morphemes
    if not ms or (ms[0].base or ms[0].surface).lower() not in _IDENTITY_PRONOUNS:
        return False
    lemmas = _lemmas_of(ms, ("名詞", "代名詞"))
    # あなたの(お)名前は / あなたは誰(ですか/なの) / Noiseとは何ですか -- all ask
    # about identity, not a word's meaning, despite sharing 何/誰 with _WHAT
    return bool(lemmas & (_IDENTITY_NOUNS | {"誰", "だれ", "何", "なに"}))


# --- the rest of _interpret's intents, same lemma/structure approach -------
_CONTENT_POS = ("名詞", "動詞", "形容詞", "感動詞", "代名詞")
_CORRECTION_LEMMAS = {"違う", "ちがう", "間違う", "間違い", "まちがい"}
_PARSE_WORD_LEMMAS = {"切り方", "区切り", "分け方", "読み方", "解析"}
_WRONG_LEMMAS = {"おかしい", "変", "違う", "間違い", "間違う"}
_FIRST_PERSON_LEMMAS = {"私", "わたし", "俺", "僕"}
_LIKE_DISLIKE_LEMMAS = {"好き", "嫌い", "好き嫌い"}
_KNOW_LEMMAS = {"知る", "覚える"}
_STATIVE_AUX_LEMMAS = {"いる", "ある"}          # 知っている/知ってる, 覚えてる
_WANT_AUX = "たい"
_CURIOSITY_OBJECT_LEMMAS = {"こと", "もの", "言葉"}
_LEARN_LEMMAS = {"学ぶ", "学習"}
_DO_LEMMA = "する"
_NOW_HERE_LEMMAS = {"今", "ここ"}
_EXIST_LEMMA = "ある"
_UNDERSTAND_LEMMAS = {"分かる", "わかる", "理解", "覚える"}


def _all_lemmas(morphemes: list) -> set[str]:
    return _lemmas_of(morphemes, _CONTENT_POS)


def _morphs(text: str):
    a = morphology.analyse(text.rstrip("。.!！？? \t\n"))
    return a.morphemes if a and a.morphemes else None


def _correction_match(text: str) -> "tuple[str, str] | None":
    """(reason, rest) if the utterance opens with a correction marker --
    "違う/ちがう/間違い/まちがい/そうじゃない" in any conjugation or politeness
    level -- or None.  `rest` is what follows the marker (and any of its own
    trailing auxiliaries/punctuation), the actual corrected content."""
    ms = _morphs(text)
    if not ms:
        m = _CORRECTION.match(text)
        return (m.group(1), m.group(2)) if m else None
    first = ms[0]
    first_lemma = first.base or first.surface
    is_marker = first.pos in ("動詞", "名詞", "形容詞") and first_lemma in _CORRECTION_LEMMAS
    # そうじゃない: 副詞[そう] + 助詞[じゃ] + 助動詞[ない] has no lemma in the set
    # above at all -- a fixed idiom, checked by its own token shape
    is_souja_nai = (len(ms) >= 3 and ms[0].surface == "そう" and ms[1].surface == "じゃ"
                    and (ms[2].base or ms[2].surface) == "ない")
    if not (is_marker or is_souja_nai):
        return None
    marker_end = 3 if is_souja_nai else 1
    reason = "そうじゃない" if is_souja_nai else first_lemma   # canonical, not raw
                                                             # conjugated surface
    end = marker_end
    # a trailing ます/です/読点 etc. is the marker's own conjugation/pause, not
    # part of the corrected content that follows
    while end < len(ms) and ms[end].pos in ("助動詞", "助詞", "記号"):
        end += 1
    rest = "".join(m.surface for m in ms[end:]).strip()
    return (reason, rest)


_PARSE_WORD_RUN = re.compile("|".join(_PARSE_WORD_LEMMAS))


def _is_parse_feedback(text: str) -> bool:
    # 切り方/区切り/分け方/読み方/解析 are compound nouns the analyser splits
    # (切り方 -> 切る + 方) with no lemma equal to the whole compound, and they
    # carry no real conjugation of their own worth generalising -- a surface
    # substring check is exactly as robust as this half needs to be.  違う/
    # おかしい/変/間違い DO conjugate (違います, おかしかった, 間違ってる), so
    # that half is still checked by lemma.
    if not _PARSE_WORD_RUN.search(text):
        return False
    ms = _morphs(text)
    if not ms:
        return bool(_PARSE_FEEDBACK.search(text))
    return bool(_all_lemmas(ms) & _WRONG_LEMMAS)


def _is_recall_preference_question(text: str) -> bool:
    ms = _morphs(text)
    if not ms:
        return bool(_RECALL_PREF.search(text))
    lemmas = _all_lemmas(ms)
    return bool(lemmas & _FIRST_PERSON_LEMMAS) and bool(lemmas & _LIKE_DISLIKE_LEMMAS) \
        and bool(lemmas & (_KNOW_LEMMAS | {"何", "なに"}))


def _wants_to_know(ms: list) -> bool:
    """助動詞 たい attached anywhere -- 知りたい/覚えたい, any conjugation."""
    return any((m.base or m.surface) == _WANT_AUX for m in ms if m.pos == "助動詞")


def _is_recall_knowledge_question(text: str) -> bool:
    """「知っている/覚えている/覚えたこと」-- a PRESENT-STATE recall, distinct
    from _is_curiosity's desire form (たい) on the very same verb lemmas."""
    ms = _morphs(text)
    if not ms:
        return bool(_RECALL.search(text))
    verb_lemmas = _lemmas_of(ms, ("動詞",))
    if not (verb_lemmas & _KNOW_LEMMAS) or _wants_to_know(ms):
        return False
    aux_lemmas = _lemmas_of(ms, ("動詞",)) - _KNOW_LEMMAS
    has_stative = bool(aux_lemmas & _STATIVE_AUX_LEMMAS)
    has_past = any((m.base or m.surface) == "た" for m in ms if m.pos == "助動詞")
    return has_stative or has_past


def _is_curiosity_question(text: str) -> bool:
    ms = _morphs(text)
    if not ms:
        return bool(_CURIOSITY.search(text))
    if _wants_to_know(ms) and (_lemmas_of(ms, ("動詞",)) & _KNOW_LEMMAS):
        return True
    lemmas = _all_lemmas(ms)
    return _wants_to_know(ms) and bool(lemmas & _CURIOSITY_OBJECT_LEMMAS)


def _is_learning_status_question(text: str) -> bool:
    ms = _morphs(text)
    if not ms:
        return bool(_LEARNING_STATUS.search(text))
    lemmas = _all_lemmas(ms)
    return bool(lemmas & _LEARN_LEMMAS) and _is_question_form(text)


def _is_activity_status_question(text: str) -> bool:
    ms = _morphs(text)
    if not ms:
        return bool(_ACTIVITY_STATUS.search(text))
    lemmas = _all_lemmas(ms)
    if _DO_LEMMA not in _lemmas_of(ms, ("動詞",)):
        return False
    return bool(lemmas & _NOW_HERE_LEMMAS) and _is_question_form(text)


def _ask_noise_preference_match(text: str) -> bool:
    ms = _morphs(text)
    if not ms:
        return bool(_ASK_NOISE_PREFERENCE.match(text))
    lemmas = _all_lemmas(ms)
    if not (lemmas & _LIKE_DISLIKE_LEMMAS) or not _is_question_form(text):
        return False
    # "Noiseの/あなたの好きな食べ物は何/ある？" -- either an explicit と/どんな
    # question word, or a leading Noise/あなた possessive makes it about
    # Noise's own preference rather than a bare "何が好き？" (ask_meaning-ish,
    # left to the generic paths)
    leads_with_noise_or_you = bool(ms) and (ms[0].base or ms[0].surface).lower() \
        in _IDENTITY_PRONOUNS
    return leads_with_noise_or_you and bool(lemmas & ({"何", "なに"} | {_EXIST_LEMMA}))


def _confirm_understanding_match(text: str) -> str | None:
    """The topic, if this asks "did you already understand X" -- any
    conjugation/politeness of 分かる/わかる/理解する/覚える, as a question."""
    ms = _morphs(text)
    if not ms:
        m = _CONFIRM.match(text)
        return _safe(m.group(1), 24) if m else None
    lemmas = _all_lemmas(ms)
    if not (lemmas & _UNDERSTAND_LEMMAS) or not _is_question_form(text):
        return None
    topic = _topic(text)
    return topic or None


def _preference_match(text: str) -> "tuple[str, str] | None":
    """(item, "好き"/"嫌い") for the user's OWN preference statement -- the
    が-marked NP immediately before a 好き/嫌い predicate, not a question."""
    ms = _morphs(text)
    if not ms:
        m = _PREFERENCE.match(text)
        return (_safe(m.group(1), 24), m.group(2)) if m else None
    if _is_question_form(text):
        return None
    value = next((("好き" if (m.base or m.surface) == "好き" else "嫌い")
                  for m in ms if m.pos == "名詞" and (m.base or m.surface) in _LIKE_DISLIKE_LEMMAS),
                 None)
    if not value:
        return None
    item_run = ""
    for m in ms:
        if m.pos == "助詞" and m.surface == "が" and item_run:
            candidate = _safe(item_run, 24)
            if candidate and candidate not in _FIRST_PERSON_LEMMAS:
                return (candidate, value)
            item_run = ""
        elif m.pos in ("名詞", "代名詞"):
            item_run += m.surface
        else:
            item_run = ""
    return None


def _what_meaning_match(text: str) -> str | None:
    """The topic, if this asks a WORD's meaning ("Xとは/って/は何ですか") --
    identity questions about Noise itself are already claimed earlier in
    _interpret, so a pronoun subject never reaches here."""
    ms = _morphs(text)
    if not ms:
        m = _WHAT.match(text)
        return _safe(m.group(1), 24) if m else None
    # 何/なに tag as pos=名詞, pos_detail=代名詞 in this analyser -- pos alone
    # (as used for the identity check above) is what actually catches them
    lemmas = _lemmas_of(ms, ("名詞", "代名詞"))
    if not ({"何", "なに"} & lemmas) or not _is_question_form(text):
        return None
    topic = _topic(text)
    return topic or None


def _claim_topic_predicate_match(text: str) -> "tuple[str, str] | None":
    """(subject, predicate) for a generic "Xは/とは Y" declarative -- the
    structural fallback below `japanese_proposition_v1.extract_proposition`
    (which only accepts a narrow, well-formed copula/existence grammar) for
    looser predicates it does not attempt.  Finds the topic-marking は
    (skipping the と of とは) via morphology instead of a regex character
    count, so a name/word containing what looks like a particle mid-string
    is not mistaken for the boundary."""
    ms = _morphs(text)
    if not ms:
        m = _CLAIM.match(text)
        return (_safe(m.group(1), 24), _safe(m.group(2))) if m else None
    topic_i = next((i for i, m in enumerate(ms)
                    if m.pos == "助詞" and m.surface == "は"), None)
    if topic_i is None or topic_i == 0:
        return None
    subject_end = topic_i - 1 if ms[topic_i - 1].surface == "と" else topic_i
    subject = _safe("".join(m.surface for m in ms[:subject_end]), 24)
    pred_ms = ms[topic_i + 1:]
    # a trailing copula (だ/です) and any sentence-final particle after it
    # (だよ/だね) are the sentence's own ending, not part of what was told --
    # matches the old regex's non-captured (?:です|だよ|だ)? suffix
    pred_end = len(pred_ms)
    while pred_end > 0 and pred_ms[pred_end - 1].pos == "助詞":
        pred_end -= 1
    if pred_end > 0 and pred_ms[pred_end - 1].pos == "助動詞" \
            and (pred_ms[pred_end - 1].base or pred_ms[pred_end - 1].surface) in ("だ", "です"):
        pred_end -= 1
    predicate = _safe("".join(m.surface for m in pred_ms[:pred_end])) or \
        _safe("".join(m.surface for m in pred_ms))
    return (subject, predicate) if subject and predicate else None


def _topic(text: str) -> str:
    quoted = _QUOTED.search(text)
    if quoted:
        return _safe(quoted.group(1), 24)
    analysis = morphology.analyse(text)
    if not analysis or not analysis.morphemes:
        explicit = _EXPLICIT_TOPIC.match(text)
        if explicit:
            candidate = _safe(explicit.group(1), 24)
            if candidate not in _TOPIC_STOP:
                return candidate
        return ""
    ms = analysis.morphemes
    # a REAL は/って topic marker is its own 助詞 token -- a string regex
    # for "って" matches its two characters wherever they occur, including
    # embedded inside an unrelated word's て-form (知って いる has "って" in
    # the middle of 知って, not as the って topic particle)
    marker_i = next((i for i, m in enumerate(ms) if m.pos == "助詞"
                     and (m.surface in ("は", "って")
                          or (m.surface == "と" and i + 1 < len(ms)
                              and ms[i + 1].surface == "は"))), None)
    if marker_i is not None and marker_i > 0:
        candidate = _safe("".join(m.surface for m in ms[:marker_i]), 24)
        if candidate and candidate not in _TOPIC_STOP:
            return candidate
    for m in ms:
        candidate = _safe(m.base or m.surface, 16)
        if not (m.pos in ("名詞", "代名詞") and candidate not in _TOPIC_STOP):
            continue
        # a single-character word is a real topic only if it is kanji/
        # katakana (猫, 犬, 山, 川) -- a lone hiragana character is never a
        # standalone noun (avoids picking up a stray okurigana/particle
        # fragment the analyser mis-split)
        if len(candidate) >= 2 or re.match(r"^[一-鿿゠-ヿ]$", candidate):
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
    # それ/これ/あれ/私/あなた have no stable referent as a STORED claim key --
    # "それは覚えることではない" must not become a literal claim keyed "それ",
    # and whose "私" it even is (the user's, or Noise's) is not resolvable
    # from the string alone.  japanese_proposition_v1 rightly allows 私 as a
    # subject for general narrative text; here it is conversational deixis.
    if prop and prop.subject not in _TOPIC_STOP:
        verbal = {"is": prop.value, "is-not": f"{prop.value}ではない",
                  "has": f"{prop.value}を持つ", "has-not": f"{prop.value}を持たない",
                  "at": f"{prop.value}にいる", "not-at": f"{prop.value}にいない"}
        return {"subject": prop.subject, "predicate": verbal[prop.relation],
                "parser": "japanese_proposition_v1", "relation": prop.relation,
                "value": prop.value}
    prop_accepted = bool(prop and prop.subject not in _TOPIC_STOP)
    if not prop_accepted and not _is_question_form(text) and not _REMARK_TAIL.search(text):
        found = _claim_topic_predicate_match(text)
        if found:
            subject, predicate = found
            # a predicate that is itself a question ("何ですか") is the user's
            # own unanswered question echoed back by a loose regex match, not
            # an answer told to Noise -- never store it as testimony
            if (subject and predicate and subject not in _TOPIC_STOP
                    and not _is_question_form(predicate)):
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
    if _is_parse_feedback(text):
        return {"intent": "parse_feedback", "confidence": 0.95,
                "topic": state.get("current_topic", ""), "detail": text}
    correction = _correction_match(text)
    if correction:
        reason, rest = correction
        return {"intent": "correction", "confidence": 0.95, "reason": reason,
                "claim": _claim_from_text(_clean(rest), state, False)}
    if _is_greeting(text):
        return {"intent": "greeting", "confidence": 1.0}
    # "あなたの好きなものは何？" shares あなた+何 with plain identity questions
    # ("Noiseとは何ですか") -- 好き/嫌い present means it is asking about a
    # PREFERENCE, not identity, so this must be checked first
    if _ask_noise_preference_match(text):
        return {"intent": "ask_noise_preference", "confidence": 0.85,
                "topic": _topic(text)}
    if _is_identity_question(text):
        return {"intent": "ask_identity", "confidence": 0.95}
    if _is_wellbeing_question(text):
        return {"intent": "ask_wellbeing", "confidence": 0.9}
    if _is_thanks(text):
        return {"intent": "thanks", "confidence": 0.95}
    if _is_capability_question(text):
        return {"intent": "ask_capability", "confidence": 0.85}
    # generic ("さっきの話に戻って") must be checked before named -- it would
    # otherwise also match _GO_BACK_NAMED with "さっき"/"前"/"元" captured as a
    # literal (and never-discussed) topic name
    if _GO_BACK_GENERIC.match(text):
        return {"intent": "go_back_topic", "confidence": 0.85, "requested_topic": ""}
    named_back = _GO_BACK_NAMED.match(text)
    if named_back:
        return {"intent": "go_back_topic", "confidence": 0.9,
                "requested_topic": _safe(named_back.group(1), 24)}
    if _FOLLOW_UP.match(text):
        return {"intent": "follow_up", "confidence": 0.85,
                "topic": state.get("current_topic", "")}
    if _is_recall_preference_question(text):
        return {"intent": "recall_preference", "confidence": 0.95}
    if _is_recall_knowledge_question(text):
        return {"intent": "recall_knowledge", "confidence": 0.95, "topic": _topic(text)}
    if _is_curiosity_question(text):
        return {"intent": "ask_curiosity", "confidence": 0.95}
    if _is_learning_status_question(text):
        return {"intent": "learning_status", "confidence": 0.9}
    if _is_activity_status_question(text):
        return {"intent": "activity_status", "confidence": 0.9}
    confirm_topic = _confirm_understanding_match(text)
    if confirm_topic:
        return {"intent": "confirm_understanding", "confidence": 0.95, "topic": confirm_topic}
    preference = _preference_match(text)
    if preference:
        return {"intent": "preference", "confidence": 0.95,
                "item": preference[0], "value": preference[1]}
    what_topic = _what_meaning_match(text)
    if what_topic:
        return {"intent": "ask_meaning", "confidence": 0.95, "topic": what_topic}
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


def _recall_prompt(state: dict, subject: str) -> str:
    """A bounded associative-recall PROPOSAL, never a claim: the nearest
    previously-discussed topic to `subject`, offered as a question the human
    confirms or denies.  Restricted to topics Noise can truthfully say it has
    actually discussed (claims heard + topic_stack); similarity alone never
    merges the two topics or writes a claim about either.  A topic usually has
    no embedding at all the first time it is ever mentioned -- there is
    nothing yet to be similar to -- so this quietly does nothing until some
    association has actually been learned, and never repeats the identical
    suggestion for the same subject twice."""
    known = (set(state.get("claims") or {}) | set(state.get("topic_stack") or [])) - {subject}
    if not known:
        return ""
    hits = association.recall(subject, state.get("embedding_memory"), known_topics=known, top_k=1)
    if not hits:
        return ""
    already = state.setdefault("topic_progress", {}).setdefault(subject, {}) \
                   .setdefault("recalled_topics", [])
    target = hits[0]["topic"]
    if target in already:
        return ""
    already.append(target)
    return f"ところで、以前「{target}」について話しましたが、関係ありますか。"


def _ask_about(state: dict, subject: str) -> str:
    count = state.setdefault("unknown_topics", {}).get(subject, 0) + 1
    state["unknown_topics"][subject] = count
    prompts = (("definition", f"{subject}はまだよく分かりません。どんなものですか。"),
               ("example", f"{subject}の具体例を一つ知りたいです。"),
               ("contrast", f"{subject}ではないものと、何が違いますか。"),
               ("use", f"{subject}は、いつ、何と一緒に現れますか。"))
    facet, reply = prompts[min(count - 1, len(prompts) - 1)]
    recall = _recall_prompt(state, subject)
    if recall:
        reply = f"{reply} {recall}"
    _touch_topic(state, subject)
    state["awaiting"] = {"kind": facet, "topic": subject, "asked_at": _now()}
    state.setdefault("question_history", {}).setdefault(subject, []).append(
        {"at": _now(), "question": reply, "attempt": count, "facet": facet})
    state["question_history"][subject] = state["question_history"][subject][-12:]
    return reply


def _belief(memory: dict | None, subject: str) -> dict:
    return ((memory or {}).get("beliefs") or {}).get(subject, {})


def _recall_knowledge(state: dict, topic: str = "") -> str:
    claims = state.get("claims", {})
    if topic and topic in claims:
        # asked about a SPECIFIC subject -- answer about that one, not every
        # subject Noise happens to hold a claim about
        live = _live_claims(state, topic)
        if not live:
            return f"「{topic}」について、あなたから聞いた説明はまだありません。"
        state["stats"]["recalls"] = state["stats"].get("recalls", 0) + 1
        heard = "、".join(f"「{c['predicate']}」" for c in live[-3:])
        return f"「{topic}」については、あなたから{heard}と聞いています。"
    if topic:
        return f"「{topic}」について、あなたから聞いた説明はまだありません。"
    memories = [f"{subject}は{live[-1]['predicate']}" for subject, claims_ in claims.items()
                if (live := [c for c in claims_ if not c.get("retracted")])]
    if not memories:
        return "あなたから聞いて覚えた説明は、まだありません。"
    state["stats"]["recalls"] = state["stats"].get("recalls", 0) + 1
    # each memory is a complete "Xは Y" statement, possibly with its own
    # internal commas -- bracket each one so the boundary between memories
    # stays unambiguous instead of running them all into one comma list
    quoted = "、".join(f"「{m}」" for m in memories[-4:])
    return f"あなたから聞いた説明では、{quoted}と覚えています。"


def _meaning_answer(state: dict, subject: str, memory: dict | None) -> str:
    live, belief = _live_claims(state, subject), _belief(memory, subject)
    _touch_topic(state, subject)
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
    _touch_topic(state, subject)
    if belief.get("understood"):
        return (f"はい。読書の用例から、{subject}を{belief.get('genus', '何か')}として扱えます。"
                f"ただし確信は{belief.get('confidence', 0):.2f}で、訂正可能です。")
    if live:
        return (f"説明は{len(live)}件覚えていますが、聞いた内容を覚えたことと理解したことは別です。"
                f"{subject}はまだ独立確認できていません。")
    return f"いいえ。{subject}について、まだ説明も独立した根拠も持っていません。"


def _follow_up_answer(state: dict, memory: dict | None) -> str:
    """「もっと教えて」「それは？」-- keep expanding on the CURRENT topic rather
    than repeating what was already said or falling back to a generic
    unresolved reply.  Surfaces one not-yet-mentioned live claim, then the
    reading belief, then rotates `_ask_about`'s facets; never repeats the
    same fact twice in a row."""
    topic = state.get("current_topic", "")
    if not topic:
        return "何について、もっと知りたいですか。話題を教えてください。"
    live, belief = _live_claims(state, topic), _belief(memory, topic)
    progress = state.setdefault("topic_progress", {}).setdefault(topic, {})
    mentioned = progress.setdefault("mentioned_predicates", [])
    unmentioned = [c for c in reversed(live) if c["predicate"] not in mentioned]
    if unmentioned:
        claim = unmentioned[0]
        mentioned.append(claim["predicate"])
        return f"{topic}については、あなたから「{claim['predicate']}」とも聞いています。"
    if belief.get("understood") and not progress.get("belief_mentioned"):
        progress["belief_mentioned"] = True
        return (f"読書では、{topic}を{belief.get('genus')}として扱っています。"
                f"根拠の強さは{belief.get('confidence', 0):.2f}です。")
    if live or belief.get("understood"):
        return f"{topic}について、今のところこれ以上お伝えできることはありません。"
    return _ask_about(state, topic)


def _go_back_answer(state: dict, requested_topic: str, memory: dict | None) -> str:
    """「さっきの話に戻って」「犬の話に戻って」-- resume an earlier topic from
    `topic_stack` (structural conversational memory), not just the single
    most recent one."""
    stack = state.get("topic_stack", [])
    if requested_topic:
        if requested_topic not in stack:
            return f"「{requested_topic}」については、まだ話していません。"
        target = requested_topic
    else:
        earlier = [t for t in stack if t != state.get("current_topic", "")]
        if not earlier:
            return "戻れる前の話題が見つかりませんでした。"
        target = earlier[-1]
    _touch_topic(state, target)
    live = _live_claims(state, target)
    belief = _belief(memory, target)
    if live:
        return (f"「{target}」の話に戻ります。あなたから「{live[-1]['predicate']}」と"
                "聞いていたところでした。")
    if belief.get("understood"):
        return (f"「{target}」の話に戻ります。読書では{belief.get('genus')}として"
                "扱っています。")
    return f"「{target}」の話に戻ります。まだ分かっていないところからでした。"


def _curiosity_answer(state: dict, memory: dict | None) -> str:
    unresolved = [(n, w) for w, n in state.get("unknown_topics", {}).items()
                  if w not in _TOPIC_STOP and not _live_claims(state, w)]
    if unresolved:
        _, topic = max(unresolved)
        _touch_topic(state, topic)
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
        _touch_topic(state, topic)
        return f"読書で出会った「{topic}」を、まだ理解できていないので知りたいです。"
    return "いまは、次に何を知るべきかを選ぶ材料が足りません。"


_SALUTATION_ECHO = {"こんにちは": "こんにちは", "こんにちわ": "こんにちは",
                    "こんばんは": "こんばんは", "こんばんわ": "こんばんは",
                    "おはよう": "おはよう"}


def _greeting(state: dict, memory: dict | None, text: str = "") -> str:
    # echo the time-of-day the USER greeted with -- replying "こんにちは"
    # (afternoon) to their "こんばんは" (evening) because the server's own
    # clock disagrees reads as ignoring what they just said.  Only やあ/
    # はじめまして (no time component) fall back to the wall clock.
    hit = _leading_interjection(_morphs(text) or [])
    lemma = (hit[0].base or hit[0].surface) if hit else ""
    salutation = _SALUTATION_ECHO.get(lemma)
    if not salutation:
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
    _touch_topic(state, head)
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


def converse(text: str, previous: dict | None, word_memory: dict | None = None,
             phraser: "PhrasingModel | None" = None) -> tuple[str, dict]:
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
        reply = _greeting(state, word_memory, text)
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
    elif intent == "follow_up":
        reply = _follow_up_answer(state, word_memory)
    elif intent == "go_back_topic":
        reply = _go_back_answer(state, interpretation.get("requested_topic", ""), word_memory)
    elif intent == "recall_preference":
        prefs = state.get("preferences", {})
        if prefs:
            reply = "あなたについては、" + "、".join(
                f"{x}が{v['preference']}" for x, v in list(prefs.items())[-3:]) + "と覚えています。"
        else:
            reply = "あなたの好き嫌いは、まだ聞いていません。"
    elif intent == "recall_knowledge":
        reply = _recall_knowledge(state, interpretation.get("topic", ""))
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
            _touch_topic(state, topic)
            reply = (f"「{topic}」の話だと思いましたが、質問なのか説明なのか決められませんでした。"
                     "質問なら「〜とは何ですか」、説明なら「〜は…です」のように言い直してもらえますか。")
        else:
            reply = ("うまく読み取れませんでした。もう少し短く、"
                     "「〜とは何ですか」のような形で聞かせてもらえますか。")

    state["dialogue_state"] = {"last_intent": intent,
                               "current_topic": state.get("current_topic", ""),
                               "awaiting": state.get("awaiting")}
    # associative recall memory: bounded, incremental, replay-safe (turn_id).
    # Learns only from claims/topic_stack Noise already holds -- never from
    # the partner's or the phrasing model's output -- and never itself writes
    # a claim; see _recall_prompt for the one place it is used.
    state["embedding_memory"] = association.learn(
        state.get("embedding_memory"), state.get("claims", {}),
        state.get("topic_stack", []), turn_id=turn_id)
    template_reply = reply
    reply = _phrase_naturally(template_reply, phraser)
    turn = {"turn_id": turn_id, "at": _now(), "user": text, "noise": reply,
            "interpretation": interpretation, "learned": learned,
            "source": "direct_human_conversation"}
    if reply != template_reply:
        turn["template_reply"] = template_reply     # audit trail: what Noise
                                                     # actually decided, before
                                                     # the presentation layer
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
            "topic_stack": list(state.get("topic_stack") or []),
            "last_intent": (state.get("dialogue_state") or {}).get("last_intent"),
            "last_exchange": state.get("last_turn")}
