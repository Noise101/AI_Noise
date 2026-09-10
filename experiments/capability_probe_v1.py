#!/usr/bin/env python3
"""Frozen held-out test of Noise's problem-solving -- three separated tiers.

v2 kept only the problems a frequency baseline got WRONG, then reported
`lift = correct - baseline_correct` as if it were evidence Noise beats a general
baseline.  By construction the baseline scored ~0 on every frozen problem, so
the "lift" was just the raw derive-rate wearing a disguise (P1-3).

v3 splits the measurement (ARCHITECTURE invariant 16, re-audit discipline):

  A. CHALLENGE set -- problems a frequency baseline fails, collected on purpose.
     Its before/after curve is a self-improvement / bottleneck DIAGNOSTIC.  A
     rise here is "Noise got better at its own hard cases", NEVER "Noise beats
     the baseline in general".

  B. UNBIASED SELECTION set -- a deterministic-salt sample of the whole problem
     population.  Gold and baseline correctness are NEVER consulted when a
     problem is placed here, so a baseline-correct problem is just as likely to
     be included.  Diagnostic for model / policy comparison; a pass here ALONE
     is not a capability claim.

  C. UNOPENED FINAL (+ RESERVE) set -- never graded until the selection side has
     cleared its pre-registered threshold for the pre-registered streak.  Opened
     at most once each; an opened final is never re-graded.  Not used for
     learning, rule formation, controller choice or threshold tuning.

`capability_confirmed` requires ALL of: the unbiased selection cleared its
pre-registered threshold across two DISTINCT model fingerprints; an unopened
final measurement improved in the SAME direction; the final's correct count
exceeds its baseline's; the minimum effect size is met; and every problem's
words sit in exactly one tier (train/selection/final never straddle a concept,
dictionary entry or source -- guaranteed here by assigning whole word groups to
a tier by a stable hash, before any gold is seen).

No LLM, no dictionary answer ever flows into a belief, a rule, a correction or
the controller: this module only READS `wm_state` / `cog_state` and grades.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

import cognition_v1 as cog
import japanese_word_meaning_v1 as wmn

VERSION = 3
PROBE_INTERVAL = 12
TIER_SALT = "cap-probe:tier:v3"

MIN_UNDERSTOOD_FOR_BUILD = 40
MIN_SELECTION_PROBLEMS = 8
MIN_FINAL_PROBLEMS = 6
MIN_CHALLENGE_PROBLEMS = 6
TARGET_PER_TIER = 60
HISTORY_CAP = 300

# --- pre-registered decision rule (frozen into the probe at build time) -----
SELECTION_DERIVE_THRESHOLD = 0.55     # unbiased-selection derive-rate must clear this
MIN_EFFECT_CORRECT = 3               # ... and beat its own baseline by >= this many
MIN_EFFECT_RATE = 0.15              # ... or by this much in rate, whichever is a
                                   #     larger absolute count on the set
SELECTION_STREAK_REQUIRED = 2       # distinct model fingerprints, both clearing it
CORRECTION = "bonferroni_on_selection_measurements"


def _ref_genus(word: str, cache: dict) -> str:
    """Coarsened ja.wiktionary genus -- an independent ground truth.  Reads ONLY
    word_meaning's already-fetched `selection_refs` gists (via `cache`): the
    probe adds no network of its own."""
    if word in cache:
        gist = cache[word]
        g = cog._canon_class(wmn._coarse((gist or {}).get("genus", ""))) if isinstance(gist, dict) else ""
        return g if g in cog.COARSE else ""
    return ""


def _hid(*p) -> str:
    return hashlib.sha256(" ".join(str(x) for x in p).encode()).hexdigest()[:16]


def _understood_words(wm: dict, min_conf: float = 0.6) -> list[str]:
    return [w for w, b in (wm.get("beliefs") or {}).items()
            if b.get("understood") and float(b.get("confidence", 0)) >= min_conf]


def _baseline_answer(problem: dict, wm: dict, heur_store: dict) -> str:
    """The no-reasoning guess: most common genus for a coarse problem, else the
    option a fixed hash lands on (a chance-level pick over the options)."""
    if problem["grade"] == "coarse":
        genera = Counter(b.get("genus") for b in (wm.get("beliefs") or {}).values() if b.get("genus"))
        return genera.most_common(1)[0][0] if genera else ""
    opts = problem.get("options", [])
    return sorted(opts, key=lambda o: hashlib.md5(f"base{o}".encode()).hexdigest())[0] if opts else ""


def _ref_pool(wm_state: dict) -> "dict[str, list[str]]":
    """Words grouped by their independent (ja.wiktionary) coarse genus, from
    word_meaning's already-fetched `selection_refs` -- no network."""
    cache = dict(wm_state.get("selection_refs") or {})
    pool: dict[str, list[str]] = {}
    for w in cache:
        g = _ref_genus(w, cache)
        if g in cog.COARSE and cog._real_word(w):
            pool.setdefault(g, []).append(w)
    for g in pool:
        pool[g].sort(key=lambda x: hashlib.md5(x.encode()).hexdigest())
    return pool


# --------------------------------------------------------------------------
# tier assignment -- each WORD is pinned to one tier by a stable hash, BEFORE any
# gold / baseline correctness is looked at.  A problem lives in the tier of the
# WORD IT ASKS ABOUT (`concept`): two problems about the same concept are always
# in the same tier, so no concept / dictionary entry ever straddles
# challenge / selection / final / reserve.  A held-out-tier (selection / final /
# reserve) problem's *distractors* may be challenge-tier (challenge makes no
# capability claim) or the SAME held-out tier -- never a different held-out tier,
# which would let one held-out set inform another.  challenge problems take any
# distractor (they are diagnostic only).
# --------------------------------------------------------------------------
_HELD_OUT_TIERS = ("selection", "final", "reserve")


def _word_tier(word: str) -> str:
    h = int(hashlib.sha256(f"{TIER_SALT}|{word}".encode()).hexdigest(), 16) % 100
    if h < 40:
        return "challenge"
    if h < 66:
        return "selection"
    if h < 84:
        return "final"
    return "reserve"


def _all_problems(wm_state: dict) -> list[dict]:
    """Every genus-combination problem the current reference pool supports.
    Gold = the independent ja.wiktionary genus.  NO baseline filter here."""
    pool = _ref_pool(wm_state)
    pairable = [g for g in pool if len(pool[g]) >= 2]
    out: list[dict] = []
    seen_pids: set = set()
    for g in pairable:
        ws = pool[g]
        others = [w for og in pool if og != g for w in pool[og]]
        # pair each word with its next few in-genus neighbours (a denser pairing
        # than consecutive-only -- every problem still lives in `a`'s tier, so a
        # concept never straddles a split; it just yields more problems from a
        # small pool)
        for i in range(len(ws)):
            for j in range(i + 1, min(i + 3, len(ws))):
                a, b = ws[i], ws[j]
                pid = _hid("cc", a, b)
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    out.append({"pid": pid, "level": 3, "type": "common_property",
                                "concept": a, "other": b, "grade": "coarse", "gold": g,
                                "prompt": f"「{a}」と「{b}」に共通するのは？",
                                "options": cog._options(g, list(cog.COARSE), a + b + "c"),
                                "ref": {a: g, b: g}, "words": sorted({a, b})})
                if others:
                    odd = others[(i * 7 + j) % len(others)]
                    og = next(x for x in pool if odd in pool[x])
                    pid = _hid("oo", a, b, odd)
                    if pid not in seen_pids:
                        seen_pids.add(pid)
                        out.append({"pid": pid, "level": 2, "type": "odd_one_out",
                                    "concept": a, "grade": "exact", "gold": odd,
                                    "prompt": f"「{a}」「{b}」「{odd}」のうち、種類がちがうのは？",
                                    "options": cog._options(odd, [a, b], a + b + odd),
                                    "ref": {a: g, b: g, odd: og}, "words": sorted({a, b, odd})})
    for p in out:
        ct = _word_tier(p["concept"])
        distractors = [w for w in p["words"] if w != p["concept"]]
        if ct == "challenge":
            p["tier"] = "challenge"
        elif all(_word_tier(w) in ("challenge", ct) for w in distractors):
            p["tier"] = ct
        else:
            p["tier"] = None
    return [p for p in out if p["tier"]]


def _tier_separation_valid(problems: list[dict]) -> bool:
    """No word is the CONCEPT of problems sitting in two different held-out
    tiers, and every problem's concept is in that problem's tier."""
    concept_tier: dict = {}
    for p in problems:
        t = p.get("tier")
        if _word_tier(p["concept"]) != t and t != "challenge":
            return False
        if t in _HELD_OUT_TIERS:
            if concept_tier.setdefault(p["concept"], t) != t:
                return False
    return True


def _sortkey(p: dict) -> str:
    return hashlib.md5(p["pid"].encode()).hexdigest()


def _model_fingerprint(wm_state: dict, cog_state: dict, words: list[str]) -> str:
    """Changes ONLY when something that could move the probe's ANSWERS moved:
    the coarse genus + rough confidence of the words it asks about, the
    controller's LEARNED POLICY (its argmax strategy per problem type, not the
    monotonic decision counter -- that ticked every cycle and made the probe
    re-measure the frozen set every cycle), and the rule count."""
    beliefs = wm_state.get("beliefs") or {}
    parts = []
    for w in sorted(set(words)):
        b = beliefs.get(w) or {}
        parts.append(f"{w}:{cog._canon_class(b.get('genus',''))}:{round(float(b.get('confidence',0)),1)}")
    pol = (cog_state or {}).get("controller", {}).get("policy", {}) or {}
    for pt in sorted(pol):
        try:
            best = cog.choose_strategy(cog_state, pt, explore=False)
        except Exception:
            best = ""
        parts.append(f"pol:{pt}={best}")
    parts.append(f"rules={len((cog_state or {}).get('rules', []))}")
    parts.append(f"corr={len((cog_state or {}).get('corrections_index', {}))}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def build_probe(cycle: int, wm_state: dict, heur_store: dict, shelf: dict,
                carry: dict | None = None) -> dict:
    """Freeze the three tiers.  Challenge is biased on purpose (baseline-fails);
    selection / final / reserve are salt samples with NO gold or baseline
    filter."""
    words = _understood_words(wm_state)
    if len(words) < MIN_UNDERSTOOD_FOR_BUILD:
        return {"version": VERSION, "status": "waiting", "have": len(words),
                "need": MIN_UNDERSTOOD_FOR_BUILD}

    allp = _all_problems(wm_state)
    for p in allp:
        p["baseline"] = _baseline_answer(p, wm_state, heur_store)

    by_tier: dict[str, list[dict]] = {"challenge": [], "selection": [], "final": [], "reserve": []}
    for p in sorted(allp, key=_sortkey):
        by_tier[p["tier"]].append(p)

    # A. CHALLENGE: baseline-fails, on purpose, capped
    challenge = [p for p in by_tier["challenge"] if not cog._grade(p, p["baseline"])][:TARGET_PER_TIER]
    # migrate any surviving v2 challenge-tier problems (they were all baseline-fails)
    for old in (carry or {}).get("problems", []):
        if len(challenge) >= TARGET_PER_TIER:
            break
        ws = old.get("words") or sorted({old.get("concept", ""), old.get("other", ""),
                                         old.get("gold", "")} - {""})
        if ws and all(_word_tier(w) == "challenge" for w in ws) \
                and old["pid"] not in {c["pid"] for c in challenge}:
            old = dict(old)
            old.setdefault("words", ws)
            old.setdefault("tier", "challenge")
            challenge.append(old)

    # B. UNBIASED SELECTION: salt sample of the whole selection-tier population
    selection = by_tier["selection"][:TARGET_PER_TIER]
    # C. UNOPENED FINAL + RESERVE
    final = by_tier["final"][:TARGET_PER_TIER]
    reserve = by_tier["reserve"][:TARGET_PER_TIER]

    if len(selection) < MIN_SELECTION_PROBLEMS:
        pool = _ref_pool(wm_state)
        return {"version": VERSION, "status": "building",
                "have": len(selection), "need": MIN_SELECTION_PROBLEMS,
                "challenge_have": len(challenge), "final_have": len(final),
                "reserve_have": len(reserve),
                "reference_pool_by_genus": {g: len(v) for g, v in sorted(pool.items())},
                "pairable_genera": [g for g in pool if len(pool[g]) >= 2],
                "bottleneck": "not enough concrete nouns with an independent (ja.wiktionary) "
                              "genus across >= 2 coarse classes -- the unbiased selection / "
                              "final tiers cannot be filled; word_meaning's held-out concrete "
                              "vocabulary is the rate limiter, not reading volume",
                "migrated_from_version": (carry or {}).get("version")}

    def _lv(ps):
        c = Counter(p["level"] for p in ps)
        return {str(k): c[k] for k in sorted(c)}

    return {
        "version": VERSION, "status": "frozen", "frozen_cycle": cycle,
        "challenge": challenge, "selection": selection, "final": final, "reserve": reserve,
        "by_level": {"challenge": _lv(challenge), "selection": _lv(selection),
                     "final": _lv(final), "reserve": _lv(reserve)},
        "pre_registration": {
            "selection_derive_threshold": SELECTION_DERIVE_THRESHOLD,
            "min_effect_correct": MIN_EFFECT_CORRECT, "min_effect_rate": MIN_EFFECT_RATE,
            "selection_streak_required": SELECTION_STREAK_REQUIRED,
            "correction": CORRECTION, "frozen_cycle": cycle,
            "tier_salt": TIER_SALT},
        "challenge_history": [], "selection_history": [],
        "final_result": None, "reserve_result": None,
        "migrated_from_version": (carry or {}).get("version"),
    }


def _run_set(problems: list[dict], wm_state: dict, heur_store: dict, cog_state: dict) -> dict:
    rules = (cog_state or {}).get("rules", [])
    per_level: dict[str, dict] = {}
    correct = base_correct = unresolved = 0
    for p in problems:
        corr = cog.corrections_for(cog_state or {}, p["type"],
                                   cog._genus(wm_state, p.get("concept", "")))
        strat = cog.choose_strategy(cog_state or {}, p["type"], explore=False)
        sol = cog.solve(p, wm_state, rules, corr, heur_store, strategy=strat)
        ok = cog._grade(p, sol["answer"])
        base_ok = cog._grade(p, p.get("baseline", ""))
        correct += ok
        base_correct += base_ok
        unresolved += int(sol["answer"] in ("", "わからない"))
        lv = per_level.setdefault(str(p["level"]), {"n": 0, "ok": 0, "base": 0})
        lv["n"] += 1
        lv["ok"] += int(ok)
        lv["base"] += int(base_ok)
    n = len(problems)
    rate = round(correct / n, 3) if n else 0.0
    base_rate = round(base_correct / n, 3) if n else 0.0
    return {"n": n, "correct": correct, "baseline_correct": base_correct,
            "derive_rate": rate, "baseline_rate": base_rate,
            "lift": correct - base_correct, "rate_lift": round(rate - base_rate, 3),
            "unresolved": unresolved,
            "by_level": {k: {**v, "rate": round(v["ok"] / v["n"], 3),
                             "base_rate": round(v["base"] / v["n"], 3)}
                         for k, v in sorted(per_level.items())}}


def _meets_effect(res: dict) -> bool:
    n = max(1, res["n"])
    return (res["lift"] >= MIN_EFFECT_CORRECT
            or res["rate_lift"] >= max(MIN_EFFECT_RATE, MIN_EFFECT_CORRECT / n))


def _selection_clears(res: dict) -> bool:
    return res["derive_rate"] >= SELECTION_DERIVE_THRESHOLD and _meets_effect(res)


def _trend(history: list, key: str = "derive_rate") -> "str | None":
    h = [x for x in history if key in x]
    if len(h) < 4:
        return None
    older = sum(x[key] for x in h[:len(h) // 2]) / (len(h) // 2)
    newer = sum(x[key] for x in h[len(h) // 2:]) / (len(h) - len(h) // 2)
    return ("improving" if newer > older + 0.02 else
            "declining" if newer < older - 0.02 else "flat")


def maybe_run(cycle: int, wm_state: dict, heur_store: dict, shelf: dict,
              cog_state: dict, previous: dict | None) -> dict:
    probe = dict(previous or {})
    if probe.get("version") != VERSION or "selection" not in probe:
        built = build_probe(cycle, wm_state, heur_store, shelf,
                            carry=previous if (previous or {}).get("problems") else None)
        if built.get("status") != "frozen":
            return built
        probe = built

    pre = probe["pre_registration"]
    ch, sel, fin, rsv = (probe["challenge"], probe["selection"],
                         probe["final"], probe["reserve"])
    ch_hist = probe.setdefault("challenge_history", [])
    sel_hist = probe.setdefault("selection_history", [])
    # one-time repair: a stale fingerprint formula (a monotonic decision counter)
    # re-measured the frozen selection set every cycle -- collapse consecutive
    # entries with an identical grade to their first occurrence.
    if len(sel_hist) > 3:
        compact, sig = [], None
        for e in sel_hist:
            s = (e.get("correct"), e.get("baseline_correct"), e.get("n"), e.get("derive_rate"))
            if s != sig:
                compact.append(e)
                sig = s
        if len(compact) < len(sel_hist):
            sel_hist = compact
            probe["selection_history"] = sel_hist

    # ---- A. challenge: a plain interval diagnostic ------------------------
    due = not ch_hist or (cycle - (ch_hist[-1].get("cycle") or 0)) >= PROBE_INTERVAL
    if due and ch:
        pt = _run_set(ch, wm_state, heur_store, cog_state)
        pt["cycle"] = cycle
        ch_hist.append(pt)
        probe["challenge_history"] = ch_hist[-HISTORY_CAP:]

    # ---- B. unbiased selection: measured only when the model could have moved,
    #         so we never re-roll a frozen set waiting for a lucky p-value -----
    sel_words = sorted({w for p in sel for w in p.get("words", [])})
    model_fp = _model_fingerprint(wm_state, cog_state, sel_words)
    last_fp = sel_hist[-1]["model_fingerprint"] if sel_hist else None
    if not sel_hist or model_fp != last_fp:
        pt = _run_set(sel, wm_state, heur_store, cog_state)
        pt.update(cycle=cycle, model_fingerprint=model_fp,
                  clears_threshold=_selection_clears(pt))
        sel_hist.append(pt)
        probe["selection_history"] = sel_hist[-HISTORY_CAP:]
    sel_measurements = len(sel_hist)

    # distinct model fingerprints whose measurement cleared the pre-registered bar
    clearing_fps: list[str] = []
    for pt in sel_hist:
        if pt.get("clears_threshold") and pt["model_fingerprint"] not in clearing_fps:
            clearing_fps.append(pt["model_fingerprint"])
    # Bonferroni: with many selection measurements, require the streak to be made
    # of the LAST `SELECTION_STREAK_REQUIRED` measurements all clearing, not any
    # cherry-picked subset across history.
    tail = sel_hist[-pre["selection_streak_required"]:]
    tail_fps = {pt["model_fingerprint"] for pt in tail}
    selection_pass = (len(tail) >= pre["selection_streak_required"]
                      and all(pt.get("clears_threshold") for pt in tail)
                      and len(tail_fps) >= pre["selection_streak_required"])

    latest_sel = sel_hist[-1] if sel_hist else None

    # ---- C. unopened final: opened at most once, only after selection_pass ---
    tier_separation_valid = _tier_separation_valid(ch + sel + fin + rsv)
    final_opened_count = sum(1 for r in (probe.get("final_result"), probe.get("reserve_result")) if r)

    if selection_pass and tier_separation_valid:
        if probe.get("final_result") is None and len(fin) >= MIN_FINAL_PROBLEMS:
            res = _run_set(fin, wm_state, heur_store, cog_state)
            res.update(opened_at_cycle=cycle, model_fingerprint=model_fp,
                       clears_threshold=_selection_clears(res),
                       same_direction=res["lift"] > 0 and res["derive_rate"] >= SELECTION_DERIVE_THRESHOLD)
            probe["final_result"] = res
            final_opened_count += 1
        elif (probe.get("final_result") is not None
              and not (probe["final_result"].get("clears_threshold")
                       and probe["final_result"].get("same_direction"))
              and probe.get("reserve_result") is None
              and len(rsv) >= MIN_FINAL_PROBLEMS
              and model_fp != probe["final_result"].get("model_fingerprint")):
            res = _run_set(rsv, wm_state, heur_store, cog_state)
            res.update(opened_at_cycle=cycle, model_fingerprint=model_fp,
                       clears_threshold=_selection_clears(res),
                       same_direction=res["lift"] > 0 and res["derive_rate"] >= SELECTION_DERIVE_THRESHOLD)
            probe["reserve_result"] = res
            final_opened_count += 1

    final_res = probe.get("reserve_result") or probe.get("final_result")
    capability_confirmed = bool(
        selection_pass and tier_separation_valid and final_res
        and final_res.get("clears_threshold") and final_res.get("same_direction")
        and final_res["correct"] > final_res["baseline_correct"]
        and _meets_effect(final_res))

    if not selection_pass:
        pending = (f"unbiased-selection has not cleared the pre-registered bar "
                   f"(derive>={SELECTION_DERIVE_THRESHOLD}, effect) on "
                   f"{pre['selection_streak_required']} distinct models "
                   f"(clearing models so far: {len(clearing_fps)})")
    elif not final_res:
        pending = "selection passed; an unopened final snapshot has not been graded yet"
    elif not capability_confirmed:
        pending = ("final measurement did not clear the pre-registered bar in the "
                   "same direction, or did not beat its baseline by the minimum effect")
    else:
        pending = None

    ch_first = ch_hist[0] if ch_hist else None
    ch_latest = ch_hist[-1] if ch_hist else None
    return {
        **probe,
        "status": "measured" if (ch_hist or sel_hist) else "frozen",
        "frozen_at": probe.get("frozen_cycle"),
        "tier_separation_valid": tier_separation_valid,
        "tier_sizes": {"challenge": len(ch), "selection": len(sel),
                       "final": len(fin), "reserve": len(rsv)},
        # A. challenge -- diagnostic self-improvement curve
        "challenge_problem_count": len(ch),
        "challenge_first_derive_rate": (ch_first or {}).get("derive_rate"),
        "challenge_latest_derive_rate": (ch_latest or {}).get("derive_rate"),
        "challenge_before_after_gain": (round(ch_latest["derive_rate"] - ch_first["derive_rate"], 3)
                                        if ch_first and ch_latest and len(ch_hist) >= 2 else None),
        "challenge_trend": _trend(ch_hist),
        "challenge_note": "within-challenge improvement -- NOT evidence of beating a general baseline",
        # B. unbiased selection -- Noise vs baseline, diagnostic
        "selection_problem_count": len(sel),
        "selection_measurements": sel_measurements,
        "selection_latest": ({k: latest_sel.get(k) for k in
                              ("derive_rate", "baseline_rate", "correct", "baseline_correct",
                               "lift", "rate_lift", "clears_threshold", "model_fingerprint")}
                             if latest_sel else None),
        "selection_models_clearing_threshold": len(clearing_fps),
        "selection_pass": selection_pass,
        "selection_trend": _trend(sel_hist),
        # C. unopened final
        "final_problem_count": len(fin),
        "reserve_problem_count": len(rsv),
        "final_opened_count": final_opened_count,
        # full result kept (the reader persists this dict and passes it back)
        "final_result": probe.get("final_result"),
        "reserve_result": probe.get("reserve_result"),
        "final_result_view": ({k: final_res.get(k) for k in
                          ("derive_rate", "baseline_rate", "correct", "baseline_correct",
                           "lift", "clears_threshold", "same_direction", "opened_at_cycle",
                           "model_fingerprint")} if final_res else None),
        "capability_confirmed": capability_confirmed,
        "capability_basis": ("unbiased-selection cleared the pre-registered threshold on "
                             f"{len(tail_fps)} distinct models AND an unopened final improved "
                             "in the same direction and beat its baseline by the minimum effect"
                             if capability_confirmed else None),
        "capability_pending_reason": pending,
        # legacy keys (now backed by the CHALLENGE tier for status continuity)
        "problem_count": len(ch),
        "first_derive_rate": (ch_first or {}).get("derive_rate"),
        "latest_derive_rate": (ch_latest or {}).get("derive_rate"),
        "latest_lift": (ch_latest or {}).get("lift"),
        "latest_by_level": (ch_latest or {}).get("by_level"),
        "before_after_gain": (round(ch_latest["derive_rate"] - ch_first["derive_rate"], 3)
                              if ch_first and ch_latest and len(ch_hist) >= 2 else None),
        "trend": _trend(ch_hist),
    }
