#!/usr/bin/env python3
"""Judge whether a story event is plausible -- from your concepts -- and revise a
concept when a real event you read looks wrong to you.

This is the predict -> fail -> self-correct loop the Japanese reading side was
missing (ARCHITECTURE: "form concepts and causal relations from few examples,
self-correct from prediction failure").  Word meaning accumulates a co-occurrence
graph + dictionary testimony; the cognition probe reads genus labels off it.
Neither one makes a falsifiable judgement about a story and revises when it is
wrong.

  1. reading a real event `(subject, verb, object)` whose subject has a genus
     Noise believes, Noise scores its PLAUSIBILITY from its own concepts:
     abstracted `(subject_genus, verb_class, object_genus)` rules, the
     subject-genus / verb-class fit, the verb-class / object-genus fit.
  2. it also scores a CORRUPTED version (the verb class -- or the object's genus
     -- swapped for a different one).  Did Noise rate the real event above the
     fake, MORE OFTEN than a plain verb/argument-frequency baseline?
  3. a real event Noise keeps rating *implausible* is the self-correction
     signal: it accumulates per subject word, and a word whose real behaviour
     keeps looking wrong under its believed genus feeds
     `japanese_word_meaning_v1` a capped penalty against that genus (one channel,
     never the sole cause of a revision -- invariant 10).

A DIAGNOSTIC only: predicting the *next verb class* from the same concepts
(secondary_diagnostic below) is near-flat on this corpus -- the same wall that
retired the surface next-event predictors.  Plausibility discrimination is the
part that carries measurable signal, exactly as `event_structure_v1`
(verb_cloze flat, event_plausibility +6.8pt) found for English.

This is NOT the retired world-model predictors: it makes no causal claim
(narrative regularity is correlational structure, invariant 6), it is gated by a
frequency baseline on a frozen source-disjoint tier, it earns a capability only
through an unopened final, and its only output into learning is capped
counter-evidence to one genus.  No LLM, no network.  AI_NOISE_JA_PREDICTION=0.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
from collections import Counter, defaultdict

import japanese_benchmark_v1 as jb
import japanese_word_meaning_v1 as wmn

try:
    from cognition_v1 import _real_word as _real_word
except Exception:                                       # pragma: no cover
    def _real_word(w: str) -> bool:
        return bool(w) and 2 <= len(w) <= 6 and wmn._is_wordlike(w)

VERSION = 1
BENCH_SALT = "ja-prediction:plausibility:v1"
EVAL_REGIME = "event_plausibility_from_concept_v1"
SIGNIFICANCE_Z = 3.0
MIN_TRAIN_STORIES = 20
MIN_TRIALS_PER_SNAPSHOT = 40
CORRUPTERS = ("verb_class_swap", "object_genus_swap")

FEEDBACK_MIN_MISSES = 5
FEEDBACK_MISS_RATIO = 1.7
FEEDBACK_MAX_PENALTY = 0.30
OUTCOME_CAP = 60
OUTCOMES_TRACKED = 4000


def enabled() -> bool:
    return os.environ.get("AI_NOISE_JA_PREDICTION") != "0"


# --- verb classes (whole-substring stems, kana + kanji) --------------------
_INGEST = ("たべ", "食べ", "のむ", "飲", "くう", "食う", "かじ", "齧", "すす", "啜", "味わ", "あじわ")
_MENTAL = ("みる", "見", "きく", "聞", "いう", "言", "はなす", "話", "こたえ", "答え",
           "たずね", "尋ね", "よぶ", "呼", "さけ", "叫", "ささやく", "なく", "泣", "ほえ", "吠",
           "わらう", "笑", "おこる", "怒", "おもう", "思", "かんがえ", "考え", "しる", "知",
           "わすれ", "忘れ", "ねむ", "眠", "ねる", "寝", "うたう", "歌", "よろこ", "喜",
           "かなし", "悲し", "おどろ", "驚", "こわがる", "恐れ", "ゆめみ", "感じ", "きめ", "決め")
_HANDLE = ("つくる", "作", "こしらえ", "もつ", "持", "つかう", "使", "なげ", "投げ", "とる", "取",
           "かう", "買", "うる", "売", "こわす", "壊す", "わる", "割る", "きる", "切", "ひらく",
           "開", "しめる", "閉", "はこぶ", "運", "おく", "置", "ひろう", "拾", "ひっぱ", "ひく",
           "引", "つかむ", "掴", "つかまえ", "捕", "にぎる", "握", "かつぐ", "担", "ならす",
           "鳴らす", "ふく", "吹", "あてる", "なでる", "撫", "たたく", "叩", "ける", "蹴",
           "おす", "押", "ぬぐ", "脱", "さす", "刺", "うつ", "打", "なげる", "そなえ", "供え")
_MOTION = ("あるく", "歩", "はしる", "走", "とぶ", "飛", "およぐ", "泳", "くる", "来る",
           "ゆく", "いく", "行", "かえる", "帰", "つく", "着", "とまる", "止ま", "すすむ", "進",
           "のぼる", "登", "おりる", "降", "でる", "出る", "はいる", "入", "うごく", "動",
           "ながれ", "流れ", "とおる", "通", "わたる", "渡", "にげる", "逃", "おう", "追", "まわる", "回")
_CHANGE = ("なる", "成る", "かわる", "変わ", "こわれ", "壊れ", "われる", "割れ", "きえ", "消え",
           "あらわれ", "現れ", "ひらく", "開く", "しまる", "閉ま", "とける", "溶け", "もえ",
           "燃え", "ふえ", "増え", "へる", "減", "そだつ", "育", "さく", "咲", "ちる", "散",
           "しぬ", "死", "うまれ", "生まれ", "はじまる", "始ま", "おわる", "終わ")
_VC_ORDER = ("motion", "mental", "handle", "ingest", "change", "other")
_VACUOUS = ("ある", "あった", "いる", "いた", "した", "する", "して", "です", "だった",
            "できる", "できた", "である", "る", "しまう", "おる", "ない")

# which verb classes a subject of each genus can plausibly do (a hand prior,
# only used as a small tie-breaker on top of the learned counts)
_GENUS_VC_FIT = {
    "生き物": {"motion", "mental", "handle", "ingest", "change", "other"},
    "人": {"motion", "mental", "handle", "ingest", "change", "other"},
    "自然物": {"motion", "change", "other"},
    "道具": {"motion", "change", "other"},
    "食べ物": {"change", "other"},
    "植物": {"change", "other"},
    "場所": {"change", "other"},
    "気持ち": {"change", "other"},
    "出来事": {"change", "other"},
}


def _has(v: str, stems: "tuple[str, ...]") -> bool:
    return any(s in v for s in stems)


def verb_class(verb: str) -> str:
    v = (verb or "").strip()
    if not v or v in _VACUOUS:
        return ""
    if _has(v, _INGEST):
        return "ingest"
    if _has(v, _MENTAL):
        return "mental"
    if _has(v, _CHANGE) and not _has(v, _HANDLE):
        return "change"
    if _has(v, _HANDLE):
        return "handle"
    if _has(v, _MOTION):
        return "motion"
    return "other"


def _canon_genus(g: str) -> str:
    return "生き物" if g == "人" else g


def _subj_genus(word: str, wm_beliefs: dict, min_conf: float = 0.55) -> str:
    b = (wm_beliefs or {}).get((word or "").strip())
    if b and b.get("understood") and float(b.get("confidence", 0.0)) >= min_conf:
        return _canon_genus(wmn._coarse(b.get("genus", "")) or b.get("genus", ""))
    return ""


def _any_genus(word: str, wm_beliefs: dict) -> str:
    """A believed genus even if not 'understood' -- used only for the object slot
    of a corruption, never for a capability gold."""
    b = (wm_beliefs or {}).get((word or "").strip())
    return _canon_genus(b.get("genus", "")) if b and b.get("genus") else ""


# --- triples over a story -------------------------------------------------
def _triples(events: list[dict], wm_beliefs: dict) -> list[dict]:
    out = []
    for e in events:
        if not isinstance(e, dict) or e.get("provenance", "heuristic_self") != "heuristic_self":
            continue
        s = (e.get("subject") or "").strip()
        vc = verb_class(e.get("verb"))
        sg = _subj_genus(s, wm_beliefs)
        if not (sg and vc):
            continue
        out.append({"subj": s, "sg": sg, "vc": vc,
                    "og": _any_genus((e.get("obj") or "").strip(), wm_beliefs)})
    return out


class PlausibilityModel:
    """Learned from the TRAINING stories only: how often each
    (subject_genus, verb_class, object_genus) / (sg, vc) / (vc, og) combination
    was actually observed.  score() is a plausibility, not a probability."""

    def __init__(self) -> None:
        self.triple: Counter = Counter()
        self.sv: Counter = Counter()
        self.vo: Counter = Counter()
        self.vc_freq: Counter = Counter()

    def fit(self, stories: list[dict], wm_beliefs: dict) -> "PlausibilityModel":
        for s in stories:
            for t in _triples(s.get("events", []), wm_beliefs):
                self.triple[(t["sg"], t["vc"], t["og"])] += 1
                self.sv[(t["sg"], t["vc"])] += 1
                if t["og"]:
                    self.vo[(t["vc"], t["og"])] += 1
                self.vc_freq[t["vc"]] += 1
        return self

    def score(self, sg: str, vc: str, og: str) -> float:
        s = 2.0 * self.triple[(sg, vc, og)] + 1.0 * self.sv[(sg, vc)] + 1.0 * self.vo[(vc, og)]
        if vc in _GENUS_VC_FIT.get(sg, set()):
            s += 0.5
        return s

    def prefers_real(self, real: tuple, fake: tuple) -> float:
        r, f = self.score(*real), self.score(*fake)
        return 1.0 if r > f else 0.5 if r == f else 0.0


class FreqBaseline:
    """C0: prefer whichever event's verb class is globally more common (and, on a
    tie, whichever object genus is more common)."""

    def __init__(self, model: PlausibilityModel) -> None:
        self.vc_freq = model.vc_freq
        self.og_freq: Counter = Counter()
        for (vc, og), n in model.vo.items():
            self.og_freq[og] += n

    def prefers_real(self, real: tuple, fake: tuple) -> float:
        rv, fv = self.vc_freq[real[1]], self.vc_freq[fake[1]]
        if rv != fv:
            return 1.0 if rv > fv else 0.0
        ro, fo = self.og_freq[real[2]], self.og_freq[fake[2]]
        return 1.0 if ro > fo else 0.5 if ro == fo else 0.0


def _corrupt(t: dict, rng: random.Random, model: PlausibilityModel) -> "list[tuple]":
    """Two genus-level corruptions of a real triple."""
    out = []
    vcw = [model.vc_freq[c] + 1 for c in _VC_ORDER]
    bad_vc = rng.choices(_VC_ORDER, weights=vcw, k=1)[0]
    for _ in range(6):
        if bad_vc != t["vc"]:
            break
        bad_vc = rng.choices(_VC_ORDER, weights=vcw, k=1)[0]
    if bad_vc != t["vc"]:
        out.append(("verb_class_swap", (t["sg"], bad_vc, t["og"])))
    if t["og"]:
        genera = [g for g in _GENUS_VC_FIT if g != t["og"]]
        bad_og = rng.choice(genera) if genera else ""
        if bad_og and bad_og != t["og"]:
            out.append(("object_genus_swap", (t["sg"], t["vc"], bad_og)))
    return out


def _checkable(t: dict) -> bool:
    """The triple carries discriminative concept structure: a non-creature
    subject, or a known-genus object.  A (creature, no object) triple is not --
    every verb class is plausible for a creature."""
    return bool(t["og"]) or t["sg"] not in ("生き物", "人")


def _discriminate(snapshot: list[dict], model: PlausibilityModel, baseline: FreqBaseline,
                  wm_beliefs: dict, seed: str) -> "dict | None":
    per: list[float] = []
    m_hit: list[float] = []
    b_hit: list[float] = []
    total = 0
    chk_m = chk_b = chk_n = uninf = 0
    for s in snapshot:
        rng = random.Random(hashlib.sha256((seed + s["url"]).encode()).hexdigest())
        mh = bh = k = 0
        for t in _triples(s.get("events", []), wm_beliefs):
            real = (t["sg"], t["vc"], t["og"])
            for _name, fake in _corrupt(t, rng, model):
                mp, bp = model.prefers_real(real, fake), baseline.prefers_real(real, fake)
                mh += mp
                bh += bp
                k += 1
                if _checkable(t):
                    chk_m += mp
                    chk_b += bp
                    chk_n += 1
                else:
                    uninf += 1
        if k < 3:
            continue
        per.append(mh / k - bh / k)
        m_hit.append(mh / k)
        b_hit.append(bh / k)
        total += k
    n = len(per)
    if n < 4 or total < MIN_TRIALS_PER_SNAPSHOT:
        return None
    gain = sum(per) / n
    var = sum((g - gain) ** 2 for g in per) / max(1, n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    z = gain / se if se > 0 else (99.0 if gain > 0 else 0.0)
    return {"n_sources": n, "n_trials": total,
            "model_accuracy": round(sum(m_hit) / n, 3),
            "baseline_accuracy": round(sum(b_hit) / n, 3),
            "discrimination_gain": round(gain, 4), "gain_z": round(z, 2),
            "significant": z >= SIGNIFICANCE_Z and gain > 0,
            # DIAGNOSTIC: of the trials, how many carried any discriminative
            # concept structure, and the model's edge on just those
            "checkable_trials": chk_n, "uninformative_trials": uninf,
            "checkable_fraction": round(chk_n / (chk_n + uninf), 3) if (chk_n + uninf) else 0.0,
            "checkable_model_acc": round(chk_m / chk_n, 3) if chk_n else None,
            "checkable_baseline_acc": round(chk_b / chk_n, 3) if chk_n else None}


# --- secondary diagnostic: next verb-class prediction (near-flat here) -----
def _next_vc_diag(snapshot: list[dict], model: PlausibilityModel, wm_beliefs: dict) -> dict:
    marg = model.vc_freq.most_common(1)[0][0] if model.vc_freq else "other"
    m = b = n = 0
    for s in snapshot:
        ts = _triples(s.get("events", []), wm_beliefs)
        for i in range(1, len(ts)):
            cand = max(_VC_ORDER, key=lambda vc: model.sv[(ts[i]["sg"], vc)]
                       + model.triple[(ts[i]["sg"], vc, "")])
            m += int(cand == ts[i]["vc"])
            b += int(marg == ts[i]["vc"])
            n += 1
    return {"n": n, "model_accuracy": round(m / n, 3) if n else None,
            "marginal_accuracy": round(b / n, 3) if n else None,
            "gain": round((m - b) / n, 4) if n else None,
            "note": "predicting the next verb class from the subject genus is near-flat "
                    "on this corpus (the same wall as surface next-event prediction); "
                    "kept as a diagnostic, NOT a capability"}


def _model_fp(wm_beliefs: dict, rules: list, train_n: int) -> str:
    gs = sorted(f"{w}:{(b or {}).get('genus','')}:{round(float((b or {}).get('confidence',0)),1)}"
                for w, b in (wm_beliefs or {}).items() if (b or {}).get("understood"))
    h = hashlib.sha256(("|".join(gs) + f"|rules={len(rules or [])}|train={train_n}").encode())
    return h.hexdigest()[:16]


# --- per-book judge / accumulate / feed back ------------------------------
def _update_outcomes(state: dict, events: list[dict], wm_beliefs: dict,
                     model: PlausibilityModel, cycle: int) -> dict:
    outc = state.setdefault("outcomes", {})
    rng = random.Random(cycle)
    plausible = implausible = 0
    for t in _triples(events, wm_beliefs):
        real = (t["sg"], t["vc"], t["og"])
        fakes = _corrupt(t, rng, model)
        if not fakes:
            continue
        ok = all(model.prefers_real(real, f) >= 0.5 for _n, f in fakes) \
            and any(model.prefers_real(real, f) == 1.0 for _n, f in fakes)
        plausible += ok
        implausible += (not ok)
        rec = outc.setdefault(t["subj"], {"assumed_genus": t["sg"], "hits": 0,
                                          "misses": 0, "last_cycle": cycle})
        rec["assumed_genus"] = t["sg"]
        rec["last_cycle"] = cycle
        if ok:
            rec["hits"] = min(OUTCOME_CAP, rec["hits"] + 1)
            rec["misses"] = max(0, rec["misses"] - 1)
        else:
            rec["misses"] = min(OUTCOME_CAP, rec["misses"] + 1)
    if len(outc) > OUTCOMES_TRACKED:
        for w in sorted(outc, key=lambda w: outc[w]["last_cycle"])[:len(outc) - OUTCOMES_TRACKED]:
            outc.pop(w, None)
    return {"events_judged": plausible + implausible,
            "rated_plausible": plausible, "rated_implausible": implausible}


def build_feedback(state: dict, cycle: int) -> dict:
    fb: dict = {}
    for w, rec in state.get("outcomes", {}).items():
        if not _real_word(w):                  # never nudge a genus for a parse fragment
            continue
        h, m = rec.get("hits", 0), rec.get("misses", 0)
        if m >= FEEDBACK_MIN_MISSES and m > FEEDBACK_MISS_RATIO * max(h, 1):
            strength = round(min(FEEDBACK_MAX_PENALTY, 0.06 * (m - FEEDBACK_MISS_RATIO * h)), 3)
            if strength > 0:
                fb[w] = {"against": rec["assumed_genus"], "strength": strength,
                         "hits": h, "misses": m, "cycle": cycle}
    return fb


def run_prediction(cycle: int, all_stories: list[dict], just_read_events: list[dict],
                   wm_beliefs: dict, rules: list, previous: dict | None) -> dict:
    previous = previous or {}
    regime_ok = previous.get("eval_regime") == EVAL_REGIME
    prev = previous if regime_ok else {}
    state = {"outcomes": dict(prev.get("outcomes", {})),
             "candidate_checkpoints": [dict(c) for c in prev.get("candidate_checkpoints", [])]}

    prev_for_tiers = dict(prev)
    prev_for_tiers.setdefault("final_history", [
        {"tier": c["tier"], "fingerprint": c.get("final_snapshot_fingerprint"),
         "preconditions": {"final_snapshot_urls": c.get("final_snapshot_urls", [])}}
        for c in prev.get("candidate_checkpoints", [])
        if c.get("tier") and c.get("final_result") is not None])
    tiers = jb.Tiers(all_stories, BENCH_SALT, prev_for_tiers, cycle=cycle)
    train = tiers.train_stories
    model = PlausibilityModel().fit(train, wm_beliefs)

    live = _update_outcomes(state, just_read_events or [], wm_beliefs, model, cycle)
    feedback = build_feedback(state, cycle)

    base = {"version": VERSION, "eval_regime": EVAL_REGIME,
            "regime_reset_from": previous.get("eval_regime") if (previous and not regime_ok) else None,
            **tiers.report_fields(),
            "selection_snapshot": tiers.selection_snapshot,
            "reserve_snapshot": tiers.reserve_snapshot,
            "final_history": tiers.final_history,
            "train_stories": len(train), "corrupters": list(CORRUPTERS),
            "live_this_cycle": live, "outcomes_tracked": len(state["outcomes"]),
            "feedback_words": len(feedback), "prediction_feedback": feedback,
            "outcomes": state["outcomes"],
            "candidate_checkpoints": state["candidate_checkpoints"],
            "learning_curve": list(prev.get("learning_curve", []))}

    if not tiers.selection_frozen or len(train) < MIN_TRAIN_STORIES:
        return {**base, "status": "insufficient_selection_stories",
                "capability_confirmed": False, "beats_baseline": False, "selection": None,
                "selection_significant_streak": 0,
                "capability_pending_reason": tiers.selection_insufficient_reason or "not_enough_train"}

    model_fp = "|".join((_model_fp(wm_beliefs, rules, len(train)), EVAL_REGIME))
    reusable = (prev.get("selection") is not None
                and prev.get("selection_model_fp") == model_fp
                and prev.get("selection_fingerprint_at_eval") == tiers.selection_fingerprint)
    if reusable:
        sel, sel_recomputed = prev["selection"], False
    else:
        sel = _discriminate(tiers.selection_snapshot, model, FreqBaseline(model),
                            wm_beliefs, BENCH_SALT + "sel")
        sel_recomputed = True
    sel_significant = bool(sel and sel["significant"])
    sel_measurements = prev.get("selection_measurements", 0) + (1 if sel_recomputed else 0)

    streak_state = jb.advance_selection_streak(
        prev.get("selection_streak"), sel_significant and sel_recomputed, len(train), model_fp)
    if sel_significant and not sel_recomputed:
        streak_state = dict(prev.get("selection_streak") or streak_state)
    sel_sig_streak = streak_state.get("streak", 0)

    candidates = state["candidate_checkpoints"]
    if jb.should_register_candidate(candidates, sel_sig_streak, len(train)) and tiers.disjoint:
        nxt = tiers.next_unopened_final()
        cand = {"registered_at_train": len(train), "registered_at_cycle": cycle,
                "model_fingerprint": model_fp, "significance_z": SIGNIFICANCE_Z,
                "selection_result": sel, "selection_fingerprint": tiers.selection_fingerprint,
                "final_result": None}
        if nxt:
            fin = _discriminate(nxt["snapshot"], model, FreqBaseline(model),
                                wm_beliefs, BENCH_SALT + "fin" + nxt["tier"])
            cand.update(tier=nxt["tier"], final_snapshot_fingerprint=nxt["fingerprint"],
                        final_snapshot_urls=nxt["urls"], final_result=fin,
                        confirmed_at=(cycle if (fin or {}).get("significant") else None))
            tiers.record_final(nxt)
        else:
            cand.update(tier=None, note="no unopened final/reserve snapshot available")
        candidates = candidates + [cand]
        state["candidate_checkpoints"] = candidates

    cap = jb.capability_view(candidates, model_fp)
    standing = candidates[-1] if candidates else None
    final_res = next((c["final_result"] for c in reversed(candidates)
                      if (c.get("final_result") or {}).get("significant")), None)
    confirmed = cap["confirmed_ever"]

    if not candidates:
        final_status = ("registerable_next_cycle" if sel_sig_streak >= 2
                        else "awaiting_selection_streak_2")
    elif standing and standing.get("final_result") is None:
        final_status = "candidate_registered_final_snapshot_unavailable"
    elif confirmed:
        final_status = ("confirmed_current_model" if cap["confirmed_current_model"]
                        else "confirmed_earlier_checkpoint")
    else:
        final_status = "final_below_threshold"

    diag = _next_vc_diag(tiers.selection_snapshot, model, wm_beliefs)
    curve = list(prev.get("learning_curve", []))
    point = {"cycle": cycle, "train_stories": len(train), "model_fp": model_fp,
             "model_accuracy": (sel or {}).get("model_accuracy"),
             "baseline_accuracy": (sel or {}).get("baseline_accuracy"),
             "discrimination_gain": (sel or {}).get("discrimination_gain"),
             "gain_z": (sel or {}).get("gain_z"), "selection_streak": sel_sig_streak,
             "next_vc_gain": diag.get("gain")}
    if not curve or curve[-1]["cycle"] != cycle:
        curve.append(point)
    curve = curve[-200:]
    tail = [c.get("discrimination_gain") or 0.0 for c in curve[-8:]]
    trend = "insufficient_data"
    if len(tail) >= 4:
        older = sum(tail[:len(tail) // 2]) / (len(tail) // 2)
        newer = sum(tail[len(tail) // 2:]) / (len(tail) - len(tail) // 2)
        trend = ("improving" if newer > older + 0.005 else
                 "declining" if newer < older - 0.005 else "flat")

    return {
        **base, "status": "measured", "recomputed": sel_recomputed,
        "selection": sel, "selection_fingerprint_at_eval": tiers.selection_fingerprint,
        "selection_model_fp": model_fp, "selection_measurements": sel_measurements,
        "selection_streak": streak_state, "selection_significant_streak": sel_sig_streak,
        "selection_next_streak_train": streak_state.get("next_streak_train"),
        "candidate_checkpoints": candidates,
        "final_opened_count": sum(1 for c in candidates if c.get("final_result") is not None),
        "final_query_budget": jb.FINAL_QUERY_BUDGET, "final_status": final_status,
        "final_result": final_res,
        "capability_confirmed": confirmed,
        "capability_confirmed_ever": cap["confirmed_ever"],
        "capability_confirmed_current_model": cap["confirmed_current_model"],
        "confirmed_checkpoints": cap["confirmed_checkpoints"],
        "beats_baseline": confirmed,
        "capability_pending_reason": (
            None if confirmed else
            "awaiting selection streak 2 (anchor-based)" if sel_sig_streak < 2 else
            "candidate registered, no unopened final snapshot" if final_status ==
                "candidate_registered_final_snapshot_unavailable" else
            "final below threshold"),
        "prediction_trend": trend, "secondary_diagnostic_next_verb_class": diag,
        "learning_curve": curve,
        "limitations": ["plausibility discrimination: does Noise rate a real "
                        "(subject_genus, verb_class, object_genus) triple above a "
                        "genus-corrupted one, MORE than a verb/argument-frequency "
                        "baseline?  SELECTION is diagnostic; capability = a candidate "
                        "checkpoint (anchor-streak 2) passing its one-shot unopened "
                        "final.  Narrative regularity, not a causal claim (invariant 6). "
                        "Next-verb-class prediction from the same concepts is near-flat "
                        "and kept only as secondary_diagnostic."],
    }


def main() -> None:                                    # pragma: no cover
    import json
    import sys
    from pathlib import Path
    runtime = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / ".local"
    events = json.loads((runtime / "reading-events.json").read_text(encoding="utf-8"))
    curr = json.loads((runtime / "reading-curriculum.json").read_text(encoding="utf-8"))
    wm = json.loads((runtime / "reading-word-meaning.json").read_text(encoding="utf-8"))
    stories = [{"url": curr.get("shelf", {}).get(bid, {}).get("url", f"book:{bid}"), "events": ev}
               for bid, ev in events.items() if len(ev) >= 3]
    rep = run_prediction(curr.get("cycle", 0), stories, [], wm.get("beliefs", {}), [], {})
    print(json.dumps({k: rep.get(k) for k in
                      ("status", "train_stories", "selection", "secondary_diagnostic_next_verb_class",
                       "final_status", "capability_confirmed", "feedback_words")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
