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

VERSION = 1
FROZEN_CONCEPTS = 12
MIN_UNDERSTOOD_TO_START = 25
STRATEGIES = ("genus", "property", "relation", "question")
PARTNER_TIMEOUT = 18
HISTORY_CAP = 200
TURNS_CAP = 60

# the reply is not parsed; approximate its content words as the kanji runs and
# katakana runs (particles / okurigana are hiragana and drop out)
_KANJI_RUN = re.compile(r"[一-鿿々]+")
_KATA_RUN = re.compile(r"[゠-ヿー]{2,}")


def _content_terms(text: str) -> set[str]:
    return set(_KANJI_RUN.findall(text)) | set(_KATA_RUN.findall(text))
_CLARIFY = ("どういう意味", "よく分から", "よくわから", "意味が分から", "もう一度",
            "理解できません", "わかりません", "何のこと", "意味不明", "説明して")
COARSE = ("生き物", "人", "植物", "食べ物", "道具", "場所", "自然物", "出来事", "気持ち")


def enabled() -> bool:
    return os.environ.get("AI_NOISE_JA_DIALOGUE") != "0"


def _canon(g: str) -> str:
    return "生き物" if g == "人" else g


def _hid(*p) -> str:
    return hashlib.sha256(" ".join(str(x) for x in p).encode()).hexdigest()[:16]


class JapanesePartner:
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

    def respond(self, utterance: str) -> str | None:
        prompt = ("日本語を学んでいる人が下の文を言いました。話題を続けて、日本語で"
                  "一〜二文で自然に応答してください。意味が分からなければ、はっきりそう"
                  "言ってください。文法の訂正はしないでください。\n"
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


def compose(concept: str, wm: dict, rules: list[dict], strategy: str) -> str:
    b = (wm.get("beliefs") or {}).get(concept, {})
    g = _canon(b.get("genus", ""))
    if not (concept and g):
        return ""
    if strategy == "genus":
        return f"「{concept}」は{g}です。"
    if strategy == "question":
        return f"「{concept}」は{g}ですか。"
    if strategy == "property":
        v = _rule_verb(rules, g)
        if v:
            return f"「{concept}」は{v}ことがあります。"
        return f"「{concept}」は{g}の なかまです。"
    if strategy == "relation":
        r = _related(wm, concept)
        if r:
            return f"「{concept}」は「{r}」と いっしょに 出てきます。"
        return f"「{concept}」は{g}です。"
    return ""


def score(utterance: str, reply: str, concept: str, wm: dict) -> dict:
    if not reply or len(reply) < 4:
        return {"understood": False, "on_topic": False, "clarification": not reply,
                "coherent": False}
    clarify = any(m in reply for m in _CLARIFY)
    heard = _content_terms(reply)
    coherent = len(heard) >= 2 or len(reply) >= 12
    b = (wm.get("beliefs") or {}).get(concept, {})
    ctx = Counter((wm.get("contexts") or {}).get(concept, {}))
    topic_terms = {concept, _canon(b.get("genus", ""))} | {x for x, _ in ctx.most_common(8)}
    topic_terms.discard("")
    # a term matches if the concept/genus/neighbour appears anywhere in a heard run
    on_topic = concept in reply or any(
        t and any(t in h or h in t for h in heard) for t in topic_terms)
    understood = coherent and on_topic and not clarify
    return {"understood": understood, "on_topic": on_topic,
            "clarification": clarify, "coherent": coherent}


def _blank() -> dict:
    return {"version": VERSION, "frozen": [], "frozen_at": None,
            "strategy_performance": {}, "practice_rotation": 0,
            "probe_rotation": 0, "probe_round": {"attempts": {}, "started_cycle": None},
            "probe_history": [], "turns": []}


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

    if not st["frozen"]:
        pool = sorted(concepts[:60], key=lambda w: hashlib.md5(w.encode()).hexdigest())
        st["frozen"] = pool[:FROZEN_CONCEPTS]
        st["frozen_at"] = cycle

    partner = partner if partner is not None else JapanesePartner()
    if not partner.available():
        return {**st, "status": "partner_unavailable",
                "overall_understood_rate": _last_rate(st)}

    is_probe = cycle % 2 == 0
    perf = st["strategy_performance"]
    turn = None

    if is_probe and st["frozen"]:
        # frozen concept, rotating; use the best strategy so far (or genus)
        c = st["frozen"][st["probe_rotation"] % len(st["frozen"])]
        st["probe_rotation"] += 1
        strat = _best_strategy(perf) or "genus"
        utt = compose(c, wm, rules, strat)
        reply = partner.respond(utt) if utt else None
        sc = score(utt, reply or "", c, wm)
        rnd = st["probe_round"]
        rnd.setdefault("started_cycle", cycle)
        rnd["attempts"][c] = int(sc["understood"])
        turn = {"kind": "probe", "concept": c, "strategy": strat, "utterance": utt,
                "reply": reply, **sc, "cycle": cycle}
        if len(rnd["attempts"]) >= len(st["frozen"]):
            ok = sum(rnd["attempts"].values())
            n = len(rnd["attempts"])
            st["probe_history"].append({"cycle": cycle, "n": n, "understood": ok,
                                        "rate": round(ok / n, 3)})
            st["probe_history"] = st["probe_history"][-HISTORY_CAP:]
            st["probe_round"] = {"attempts": {}, "started_cycle": None}
    else:
        c = concepts[st["practice_rotation"] % len(concepts)]
        st["practice_rotation"] += 1
        strat = STRATEGIES[st["practice_rotation"] % len(STRATEGIES)]
        utt = compose(c, wm, rules, strat)
        reply = partner.respond(utt) if utt else None
        sc = score(utt, reply or "", c, wm)
        bucket = perf.setdefault(strat, {"turns": 0, "understood": 0})
        bucket["turns"] += 1
        bucket["understood"] += int(sc["understood"])
        bucket["rate"] = _beta(bucket["understood"], bucket["turns"])
        turn = {"kind": "practice", "concept": c, "strategy": strat, "utterance": utt,
                "reply": reply, **sc, "cycle": cycle}

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
    return {**st, "status": "ran",
            "frozen_count": len(st["frozen"]),
            "strategy_performance": perf, "best_strategy": _best_strategy(perf),
            "practice_turns": sum(v["turns"] for v in perf.values()),
            "probe_rounds": len(h),
            "first_understood_rate": first["rate"] if first else None,
            "overall_understood_rate": latest["rate"] if latest else None,
            "before_after_gain": (round(latest["rate"] - first["rate"], 3)
                                  if first and latest and len(h) >= 2 else None),
            "trend": trend,
            "sample_turn": turn,
            "note": "the partner is an environment; its replies never update a "
                    "belief -- only 'was I understood' is scored (invariant 13)"}


def _best_strategy(perf: dict) -> "str | None":
    ranked = [(v.get("rate", 0.0), k) for k, v in perf.items() if v.get("turns", 0) >= 4]
    return max(ranked)[1] if ranked else None


def _last_rate(st: dict):
    h = st.get("probe_history", [])
    return h[-1]["rate"] if h else None
