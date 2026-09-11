#!/usr/bin/env python3
"""Say something in Japanese from what reading taught you, and learn whether it
landed.

The English `generative_dialogue_v1` is stuck at ~6% because its utterances come
from a plateaued English vocabulary.  This module composes a Japanese utterance
from Noise's OWN reading knowledge -- a word-meaning belief, an abstracted rule,
a co-occurrence -- sends it to the local model as an *environment* (invariant 13,
not a teacher), and scores itself behaviourally on whether the reply shows
comprehension.  It is also a use-test: a belief Noise cannot put into a sentence
that a person understands is not yet usable knowledge.

Capability: a frozen set of concepts.  Every cycle one utterance is attempted --
alternating a rotating *practice* concept (updates strategy_performance) and a
frozen *probe* concept (rotating; a full round = len(frozen) cycles, and the
round's understood-rate is the tracked signal).  No LLM output ever updates a
belief.

Toggle with AI_NOISE_JA_DIALOGUE=0.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from collections import Counter

VERSION = 4                     # v4: word_meaning._norm fixed a bug that let a
                                # topic/case particle glued to a following noun
                                # ("も気", "も返事" -- really 気/返事 with a
                                # stray leading も) be believed as its own
                                # "understood" word; two of this frozen set's
                                # 12 concepts were exactly that, permanently
                                # capping the probe on non-words no amount of
                                # reading could ever fix.  Re-freezes clean.
                                # v3 (P1-2): echo-aware scoring, per-intent required
                                # response, utterance-quality split, frozen tasks
FROZEN_CONCEPTS = 12
MIN_UNDERSTOOD_TO_START = 25
STRATEGIES = ("genus", "property", "relation", "question")
PARTNER_TIMEOUT = None      # no artificial cap -- wait for the reply, however long
# run_practice calls the partner at most once per cycle, so a slower/larger
# model costs one long wait per cycle, not a loop of them.  The owner's
# standing choice is qwen3.8:27b, accepting its ~55s replies; an earlier
# revision added a 30s cap that was never the owner's decision, and it
# silently scored two whole probe rounds 0.0 by cutting the model off before
# it could answer.  The call now blocks until the model actually replies or
# the connection itself fails, instead of being cut off on a clock. Override
# with AI_NOISE_JA_DIALOGUE_MODEL if a different partner model is ever wanted.
PARTNER_MODEL = os.environ.get("AI_NOISE_JA_DIALOGUE_MODEL", "qwen3.8:27b")
HISTORY_CAP = 200
TURNS_CAP = 60

# the reply is not parsed; approximate its content words as the kanji runs and
# katakana runs (particles / okurigana are hiragana and drop out)
_KANJI_RUN = re.compile(r"[一-鿿々]+")
_KATA_RUN = re.compile(r"[゠-ヿー]{2,}")


def _content_terms(text: str) -> set[str]:
    return set(_KANJI_RUN.findall(text)) | set(_KATA_RUN.findall(text))


def _strip_marks(text: str) -> str:
    return re.sub(r"[「」、。　\s]", "", text or "")
_CLARIFY = ("どういう意味", "よく分から", "よくわから", "意味が分から", "もう一度",
            "理解できません", "わかりません", "わかりかね", "何のこと", "意味不明",
            "説明して", "教えてください", "もう少し詳しく", "具体的に")
# bare agreement / acknowledgement -- carries no independent semantic content
_AGREE = ("そうですね", "そうです", "その通り", "そのとおり", "おっしゃる通り",
          "はい、", "はい。", "ええ", "なるほど", "了解", "わかりました", "分かりました",
          "承知", "確かに", "たしかに")
COARSE = ("生き物", "人", "身体", "植物", "食べ物", "道具", "場所", "自然物", "出来事", "気持ち")
_POLARITY = ("はい", "いいえ", "ちがい", "違い", "ではない", "ではありません",
             "じゃない", "ありません", "そうです", "その通り", "正しい", "正解")
# cues that a reply is actually SAYING SOMETHING about the concept's nature
# (a use, how it is made, what it is for, its kind) rather than just naming it
_DESCRIBES = ("使", "作", "つく", "ため", "役", "種類", "一種", "仲間", "なかま",
              "できて", "材料", "とは", "意味", "食べ", "住", "生え", "咲",
              "場所", "所です", "どうぐ", "どうぶつ", "しょくぶつ", "たべもの")
# assert "<concept> IS <other coarse genus>" -- a real contradiction, not a
# passing mention of a person/place noun
_ASSERT_SUFFIX = ("です", "だ", "である", "の一種", "の仲間", "のなかま",
                  "だと思", "になり", "と言え", "と言われ")


def enabled() -> bool:
    return os.environ.get("AI_NOISE_JA_DIALOGUE") != "0"


def _canon(g: str) -> str:
    return "生き物" if g == "人" else g


def _hid(*p) -> str:
    return hashlib.sha256(" ".join(str(x) for x in p).encode()).hexdigest()[:16]


class JapanesePartner:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model or PARTNER_MODEL

    def available(self) -> bool:
        if os.environ.get("AI_NOISE_SKIP_LOCAL_LLM"):
            return False
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=2) as r:
                return any(m.get("name") == self.model for m in json.load(r).get("models", []))
        except Exception:
            return False

    def respond(self, utterance: str) -> str | None:
        prompt = ("日本語を学んでいる人が下の文を言いました。話題を続けて、必ず日本語だけで"
                  "一〜二文で自然に応答してください。英語は使わないでください。意味が分から"
                  "なければ「よく分かりません」と言ってください。文法の訂正はしないでください。\n"
                  f"学習者: {utterance[:300]}")
        schema = {"type": "object", "properties": {"reply": {"type": "string"}},
                  "required": ["reply"]}
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                              "think": False, "format": schema,
                              "options": {"temperature": 0.4, "num_predict": 80}}).encode()
        req = urllib.request.Request(f"{self.base_url}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=PARTNER_TIMEOUT) as r:
                parsed = json.loads(json.load(r).get("response", "{}"))
            return str(parsed.get("reply", "")).strip()[:400] or None
        except Exception:
            return None


# --------------------------------------------------------------------------
try:
    from cognition_v1 import _real_word, _informative_verb
except Exception:                                        # pragma: no cover
    def _real_word(w):  # noqa
        return bool(w) and 2 <= len(w) <= 6

    def _informative_verb(v):  # noqa
        return bool(v) and len(v) >= 2


def _understood_concepts(wm: dict, concrete_first: bool = True) -> list[str]:
    """Understood, real-word concepts.  Concrete genera first (talking about
    「椅子」/「病院」 is a real use-test; 「親方 は 生き物 です」 is trivially true)."""
    out = []
    for w, b in (wm.get("beliefs") or {}).items():
        if b.get("understood") and b.get("genus") and _real_word(w) \
                and float(b.get("confidence", 0)) >= 0.55:
            out.append(w)
    ent = wm.get("entities") or {}
    beliefs = wm.get("beliefs") or {}
    out.sort(key=lambda w: ((_canon(beliefs[w]["genus"]) == "生き物") if concrete_first else 0,
                            -ent.get(w, 0)))
    return out


def _related(wm: dict, concept: str) -> str:
    """A NOUN Noise has read alongside `concept` and also understands."""
    ctx = Counter((wm.get("contexts") or {}).get(concept, {}))
    own_verbs = set((wm.get("profiles") or {}).get(concept, {}).get("subj_verbs", {})) | \
        set((wm.get("profiles") or {}).get(concept, {}).get("obj_verbs", {}))
    for n, _ in ctx.most_common(15):
        b = (wm.get("beliefs") or {}).get(n)
        if n != concept and b and b.get("understood") and _real_word(n) and n not in own_verbs:
            return n
    return ""


def _rule_verb(rules: list[dict], genus: str) -> str:
    for r in rules:
        if r.get("status") == "reusable" and _canon(r.get("subject_genus", "")) == _canon(genus) \
                and not r.get("object_genus") and _informative_verb(r.get("verb", "")):
            return r["verb"]
    return ""


def _claim(concept: str, wm: dict, rules: list[dict], strategy: str) -> dict:
    """The semantic content Noise is trying to convey, plus an honest read of
    whether its OWN knowledge actually supports that utterance (P1-2 point 8).
    `text` is the sentence sent to the partner; the rest is never sent."""
    b = (wm.get("beliefs") or {}).get(concept, {})
    g = _canon(b.get("genus", ""))
    conf = float(b.get("confidence", 0.0))
    belief_supported = bool(b.get("understood") and g and conf >= 0.55 and _real_word(concept))
    cl = {"form": strategy, "concept": concept, "genus": g, "relation": "", "verb": "",
          "parseable": bool(concept and g), "belief_supported": belief_supported,
          "relation_supported": False, "text": ""}
    if not (concept and g):
        cl["malformed"] = True
        return cl
    if strategy == "genus":
        cl["text"] = f"「{concept}」は{g}です。"
    elif strategy == "question":
        cl["text"] = f"「{concept}」は{g}ですか。"
    elif strategy == "property":
        v = _rule_verb(rules, g)
        if v:
            cl["verb"] = v
            cl["text"] = f"「{concept}」は{v}ことがあります。"
        else:
            cl["text"] = f"「{concept}」は{g}の なかまです。"
    elif strategy == "relation":
        r = _related(wm, concept)
        if r:
            cl["relation"], cl["relation_supported"] = r, True
            cl["text"] = f"「{concept}」は「{r}」と いっしょに 出てきます。"
        else:
            cl["text"] = f"「{concept}」は{g}です。"          # degraded to a genus claim
    cl["malformed"] = not belief_supported or (strategy == "relation" and not cl["relation_supported"])
    return cl


def compose(concept: str, wm: dict, rules: list[dict], strategy: str) -> str:
    return _claim(concept, wm, rules, strategy)["text"]


def score(claim: dict, reply: str, wm: dict) -> dict:
    """Behavioural comprehension score (invariant 13: the partner is an
    environment, its reply is never a fact and never touches a belief).

    A reply passes ONLY if, after removing everything it copied from Noise's own
    sentence, it still carries independent semantic content AND that content is
    the KIND of response the utterance's form asks for.  A verbatim / partial
    echo, a bare "そうですね", a helpful guess at a malformed sentence, and an
    off-topic reply all fail -- each for a recorded reason."""
    concept = claim.get("concept", "")
    form = claim.get("form", "")
    g = claim.get("genus", "")
    said_terms = _content_terms(claim.get("text", "")) | {concept, g, claim.get("relation", "")}
    said_terms.discard("")

    base = {"echo_detected": False, "clarification": False, "coherent": False,
            "on_topic": False, "relevant_new_information": False,
            "response_to_requested_act": False, "contradicts_belief": False,
            "partner_guessed_malformed": False, "understood": False,
            "reply_chars": len(reply or "")}
    if not reply or len(_strip_marks(reply)) < 4:
        base["clarification"] = not reply
        return base

    clarify = any(m in reply for m in _CLARIFY)
    heard = _content_terms(reply)
    base["coherent"] = len(heard) >= 2 or len(_strip_marks(reply)) >= 12
    base["clarification"] = clarify

    # --- echo detection --------------------------------------------------
    core = _strip_marks(claim.get("text", ""))
    for tail in ("ですか", "です", "ことがあります", "のなかまです", "といっしょに出てきます"):
        core = core[: -len(tail)] if core.endswith(tail) else core
    rep_stripped = _strip_marks(reply)
    verbatim = core and (core in rep_stripped or claim.get("text", "").strip() in reply
                         or "学習者" in reply)
    # of the reply's content runs, how many are things Noise already said?
    novel = {h for h in heard
             if not any(h == s or h in s or s in h for s in said_terms if s)}
    echo_share = 1.0 - (len(novel) / len(heard)) if heard else 1.0
    base["echo_detected"] = bool(verbatim or (echo_share >= 0.75 and len(rep_stripped) <= len(core) + 8))

    # --- independent new information -----------------------------------
    # strip everything Noise said, plus agreement markers, and see what remains
    residue = reply
    for s in sorted(said_terms, key=len, reverse=True):
        if s:
            residue = residue.replace(s, "")
    for a in _AGREE + _CLARIFY:
        residue = residue.replace(a, "")
    residue_content = _content_terms(residue)
    base["relevant_new_information"] = (
        not base["echo_detected"] and len(residue_content) >= 1
        and len(_strip_marks(residue)) >= 6)

    # --- on topic: naming the concept is NOT enough (P1-2 point 4) -----
    ctx = Counter((wm.get("contexts") or {}).get(concept, {}))
    neighbours = {x for x, _ in ctx.most_common(8) if x}
    names_concept = concept in reply
    mentions_genus = bool(g) and g in reply
    mentions_neighbour = any(nb in reply for nb in neighbours)
    describes = any(cue in reply for cue in _DESCRIBES)
    r = claim.get("relation", "")
    base["on_topic"] = bool(
        mentions_genus or mentions_neighbour
        or (names_concept and (describes or (r and r in reply)))
        or (r and r in reply and describes))

    # --- contradiction: the reply ASSERTS a different coarse genus ----
    other_asserted = [c for c in COARSE if _canon(c) != g
                      and any(f"{c}{suf}" in reply for suf in _ASSERT_SUFFIX)]
    base["contradicts_belief"] = bool(other_asserted) and form != "question"

    # --- the response the utterance's form asks for ------------------
    if form == "question":
        base["response_to_requested_act"] = (
            any(p in reply for p in _POLARITY) or any(c in reply for c in COARSE))
    elif form == "relation":
        r = claim.get("relation", "")
        # must say something about the pair beyond restating the co-occurrence
        base["response_to_requested_act"] = (
            base["relevant_new_information"]
            and (concept in reply or (r and r in reply)))
    else:                       # genus / property -> a use, attribute, hypernym, example
        base["response_to_requested_act"] = base["relevant_new_information"] and base["on_topic"]

    # --- a helpful guess at a sentence Noise could not ground --------
    base["partner_guessed_malformed"] = bool(claim.get("malformed"))

    base["understood"] = bool(
        base["coherent"] and base["on_topic"] and base["relevant_new_information"]
        and base["response_to_requested_act"]
        and not base["echo_detected"] and not clarify
        and not base["contradicts_belief"]
        and not claim.get("malformed"))
    return base


def _blank() -> dict:
    return {"version": VERSION, "frozen": [], "frozen_tasks": [], "frozen_at": None,
            "strategy_performance": {}, "practice_rotation": 0,
            "probe_rotation": 0, "probe_round": {"attempts": {}, "started_cycle": None},
            "probe_history": [], "turns": []}


_QUALITY_KEYS = ("parseable", "belief_supported", "relation_supported", "malformed")
_BEHAVIOUR_KEYS = ("echo_detected", "clarification", "relevant_new_information",
                   "response_to_requested_act", "contradicts_belief",
                   "partner_guessed_malformed", "on_topic", "coherent", "understood")


def _frozen_strategy(concept: str) -> str:
    """A FIXED utterance intent per frozen concept (P1-2 point 9): the probe
    freezes the task, not just the word, so a shift in the learned best strategy
    cannot move the frozen metric."""
    h = int(hashlib.md5(("task:" + concept).encode()).hexdigest()[:8], 16)
    return STRATEGIES[h % len(STRATEGIES)]


def _beta(ok: int, n: int) -> float:
    return round((ok + 1) / (n + 2), 4)


def run_practice(wm: dict, rules: list[dict], previous: dict | None,
                 cycle: int, partner: "JapanesePartner | None" = None) -> dict:
    st = dict(previous or _blank())
    for k, v in _blank().items():
        st.setdefault(k, v)
    if st.get("version") != VERSION:
        st = _blank()

    concepts = _understood_concepts(wm)
    if len(concepts) < MIN_UNDERSTOOD_TO_START:
        return {**st, "status": "waiting", "have": len(concepts), "need": MIN_UNDERSTOOD_TO_START}

    if not st.get("frozen_tasks"):
        clean = [w for w in concepts
                 if " " not in w and "　" not in w and 2 <= len(w) <= 5
                 and not w.startswith(("一", "何", "美", "自分", "そう", "こう"))]
        pool = sorted((clean or concepts)[:60], key=lambda w: hashlib.md5(w.encode()).hexdigest())
        picked = pool[:FROZEN_CONCEPTS]
        st["frozen"] = picked
        st["frozen_tasks"] = [{"concept": c, "strategy": _frozen_strategy(c)} for c in picked]
        st["frozen_at"] = cycle

    partner = partner if partner is not None else JapanesePartner()
    if not partner.available():
        return {**st, "status": "partner_unavailable",
                "overall_understood_rate": _last_rate(st)}

    is_probe = cycle % 2 == 0
    perf = st["strategy_performance"]
    turn = None
    tasks = st["frozen_tasks"]

    def _attempt(c: str, strat: str, kind: str) -> dict:
        cl = _claim(c, wm, rules, strat)
        utt = cl["text"]
        reply = partner.respond(utt) if utt else None
        sc = score(cl, reply or "", wm)
        quality = {k: cl.get(k, False) for k in _QUALITY_KEYS}
        return {"kind": kind, "concept": c, "strategy": strat, "utterance": utt,
                "reply": reply, **sc, "quality": quality, **quality, "cycle": cycle}

    if is_probe and tasks:
        task = tasks[st["probe_rotation"] % len(tasks)]
        st["probe_rotation"] += 1
        c, strat = task["concept"], task["strategy"]           # BOTH frozen
        turn = _attempt(c, strat, "probe")
        rnd = st["probe_round"]
        rnd.setdefault("started_cycle", cycle)
        rnd["attempts"][c] = int(turn["understood"])
        if len(rnd["attempts"]) >= len(tasks):
            ok = sum(rnd["attempts"].values())
            n = len(rnd["attempts"])
            recent = [t for t in (st["turns"] + [turn])[-n:] if t.get("kind") == "probe"]
            m = max(1, len(recent))
            st["probe_history"].append({
                "cycle": cycle, "n": n, "understood": ok, "rate": round(ok / n, 3),
                "echo_rate": round(sum(1 for t in recent if t.get("echo_detected")) / m, 3),
                "clarification_rate": round(sum(1 for t in recent if t.get("clarification")) / m, 3),
                "malformed_rate": round(sum(1 for t in recent if t.get("malformed")) / m, 3),
                "new_info_rate": round(sum(1 for t in recent if t.get("relevant_new_information")) / m, 3)})
            st["probe_history"] = st["probe_history"][-HISTORY_CAP:]
            st["probe_round"] = {"attempts": {}, "started_cycle": None}
    else:
        c = concepts[st["practice_rotation"] % len(concepts)]
        st["practice_rotation"] += 1
        strat = STRATEGIES[st["practice_rotation"] % len(STRATEGIES)]
        turn = _attempt(c, strat, "practice")
        bucket = perf.setdefault(strat, {"turns": 0, "understood": 0})
        bucket["turns"] += 1
        bucket["understood"] += int(turn["understood"])
        bucket["rate"] = _beta(bucket["understood"], bucket["turns"])

    if turn:
        st["turns"] = (st["turns"] + [turn])[-TURNS_CAP:]

    h = st["probe_history"]
    first, latest = (h[0] if h else None), (h[-1] if h else None)
    trend = None
    if len(h) >= 4:
        older = sum(x["rate"] for x in h[:len(h) // 2]) / (len(h) // 2)
        newer = sum(x["rate"] for x in h[len(h) // 2:]) / (len(h) - len(h) // 2)
        trend = ("improving" if newer > older + 0.03 else
                 "declining" if newer < older - 0.03 else "flat")
    recent_probe = [t for t in st["turns"] if t.get("kind") == "probe"][-len(tasks or [1]) * 2:]
    def _rate(key):
        return (round(sum(1 for t in recent_probe if t.get(key)) / len(recent_probe), 3)
                if recent_probe else None)
    return {**st, "status": "ran",
            "frozen_count": len(tasks),
            "frozen_tasks": tasks,
            "strategy_performance": perf, "best_strategy": _best_strategy(perf),
            "practice_turns": sum(v["turns"] for v in perf.values()),
            "probe_rounds": len(h),
            "first_understood_rate": first["rate"] if first else None,
            "overall_understood_rate": latest["rate"] if latest else None,
            "before_after_gain": (round(latest["rate"] - first["rate"], 3)
                                  if first and latest and len(h) >= 2 else None),
            "trend": trend,
            # behavioural + utterance-quality breakdown over the recent probe turns
            "recent_echo_rate": _rate("echo_detected"),
            "recent_clarification_rate": _rate("clarification"),
            "recent_new_info_rate": _rate("relevant_new_information"),
            "recent_requested_act_rate": _rate("response_to_requested_act"),
            "recent_malformed_rate": _rate("malformed"),
            "recent_belief_supported_rate": _rate("belief_supported"),
            "latest_round_metrics": {k: latest.get(k) for k in
                ("echo_rate", "clarification_rate", "malformed_rate", "new_info_rate")}
                if latest else None,
            "sample_turn": turn,
            "note": "the partner is an environment; its replies never update a "
                    "belief -- only 'was I understood' is scored (invariant 13).  "
                    "A verbatim/partial echo, a bare agreement, a helpful guess at a "
                    "malformed utterance, and an off-topic reply all fail."}


def _best_strategy(perf: dict) -> "str | None":
    ranked = [(v.get("rate", 0.0), k) for k, v in perf.items() if v.get("turns", 0) >= 4]
    return max(ranked)[1] if ranked else None


def _last_rate(st: dict):
    h = st.get("probe_history", [])
    return h[-1]["rate"] if h else None
