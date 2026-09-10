#!/usr/bin/env python3
"""Frozen held-out test of Noise's problem-solving, not its knowledge count.

The English pipeline plateaued because "more curricula" was measured by stored
knowledge, not by a frozen transfer test (ARCHITECTURE invariant 16).  This
module is built first, before the cognitive loop is trusted:

  * `build_probe` freezes ~40 problems the day the loop turns on, with the gold
    answers derived from Noise's own held-out-validated beliefs, the event order
    of specific books, and the coarse ontology.  The set is NEVER regenerated
    and NEVER fed to abstraction.
  * every PROBE_INTERVAL cycles the current cognitive `solve` re-attempts all
    frozen problems WITHOUT the gold.  We record the derive-rate and the lift
    over a per-problem frequency baseline, per level.

"Noise reads more" is not progress.  "The frozen derive-rate rises while the
problem set is fixed" is.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import cognition_v1 as cog
import japanese_word_meaning_v1 as wmn

VERSION = 2
PROBE_INTERVAL = 12
TARGET_PROBLEMS = 40
MIN_PROBE_PROBLEMS = 6           # freeze once this many reference-checked problems exist
MIN_UNDERSTOOD_FOR_BUILD = 40
HISTORY_CAP = 300


def _ref_genus(word: str, cache: dict) -> str:
    """Coarsened ja.wiktionary genus -- an independent ground truth.  Reads
    ONLY word_meaning's already-fetched `selection_refs` gists (via `cache`,
    which is that dict): the probe adds no network of its own."""
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
    """Words grouped by their independent (ja.wiktionary) coarse genus, drawn
    from word_meaning's already-fetched `selection_refs` -- no network."""
    cache = dict(wm_state.get("selection_refs") or {})
    pool: dict[str, list[str]] = {}
    for w in cache:
        g = _ref_genus(w, cache)
        if g in cog.COARSE and cog._real_word(w):
            pool.setdefault(g, []).append(w)
    for g in pool:
        pool[g].sort(key=lambda x: hashlib.md5(x.encode()).hexdigest())
    return pool


def build_probe(cycle: int, wm_state: dict, heur_store: dict, shelf: dict) -> dict:
    """Freeze genus-combination problems whose GOLD is the independent
    ja.wiktionary genus (not Noise's own belief): the score then measures
    whether belief-based reasoning agrees with an outside reference over time."""
    words = _understood_words(wm_state)
    if len(words) < MIN_UNDERSTOOD_FOR_BUILD:
        return {"version": VERSION, "status": "waiting", "have": len(words),
                "need": MIN_UNDERSTOOD_FOR_BUILD}
    pool = _ref_pool(wm_state)
    pairable = [g for g in pool if len(pool[g]) >= 2]
    problems: list[dict] = []
    for g in pairable:
        ws = pool[g]
        others = [w for og in pool if og != g for w in pool[og]]
        for i in range(0, len(ws) - 1, 2):
            a, b = ws[i], ws[i + 1]
            p = {"pid": _hid("cc", a, b), "level": 3, "type": "common_property",
                 "concept": a, "other": b, "grade": "coarse", "gold": g,
                 "prompt": f"「{a}」と「{b}」に共通するのは？",
                 "options": cog._options(g, list(cog.COARSE), a + b + "c"),
                 "ref": {a: g, b: g}}
            p["baseline"] = _baseline_answer(p, wm_state, heur_store)
            if not cog._grade(p, p["baseline"]):
                problems.append(p)
            if others:
                odd = others[(i // 2) % len(others)]
                og = next(x for x in pool if odd in pool[x])
                q = {"pid": _hid("oo", a, b, odd), "level": 2, "type": "odd_one_out",
                     "concept": a, "grade": "exact", "gold": odd,
                     "prompt": f"「{a}」「{b}」「{odd}」のうち、種類がちがうのは？",
                     "options": cog._options(odd, [a, b], a + b + odd),
                     "ref": {a: g, b: g, odd: og}}
                q["baseline"] = _baseline_answer(q, wm_state, heur_store)
                if not cog._grade(q, q["baseline"]):
                    problems.append(q)
    problems.sort(key=lambda p: hashlib.md5(p["pid"].encode()).hexdigest())
    problems = problems[:TARGET_PROBLEMS]
    if len(problems) < MIN_PROBE_PROBLEMS:
        return {"version": VERSION, "status": "building", "have": len(problems),
                "need": MIN_PROBE_PROBLEMS}
    by_level = Counter(p["level"] for p in problems)
    return {"version": VERSION, "status": "frozen", "frozen_cycle": cycle,
            "problems": problems, "by_level": {str(k): by_level[k] for k in sorted(by_level)},
            "history": []}


def run_probe(probe: dict, wm_state: dict, heur_store: dict, cog_state: dict) -> dict:
    rules = cog_state.get("rules", [])
    per_level: dict[str, dict] = {}
    correct = base_correct = 0
    unresolved = 0
    for p in probe.get("problems", []):
        corr = cog.corrections_for(cog_state, p["type"], cog._genus(wm_state, p.get("concept", "")))
        # the frozen probe reflects the CONTROLLER's learned policy (no
        # exploration -- salt fixed) so the metric tracks what the loop settled on
        strat = cog.choose_strategy(cog_state, p["type"], explore=False)
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
    n = len(probe.get("problems", []))
    point = {"cycle": None, "n": n, "correct": correct, "baseline": base_correct,
             "derive_rate": round(correct / n, 3) if n else 0.0,
             "lift": correct - base_correct, "unresolved": unresolved,
             "by_level": {k: {**v, "rate": round(v["ok"] / v["n"], 3),
                              "base_rate": round(v["base"] / v["n"], 3)}
                          for k, v in sorted(per_level.items())}}
    return point


def maybe_run(cycle: int, wm_state: dict, heur_store: dict, shelf: dict,
              cog_state: dict, previous: dict | None) -> dict:
    probe = dict(previous or {})
    if probe.get("version") != VERSION or "problems" not in probe:
        built = build_probe(cycle, wm_state, heur_store, shelf)
        if built.get("status") != "frozen":
            return built
        probe = built
    last = probe.get("history", [])
    due = not last or (cycle - (last[-1].get("cycle") or 0)) >= PROBE_INTERVAL
    if due:
        point = run_probe(probe, wm_state, heur_store, cog_state)
        point["cycle"] = cycle
        probe.setdefault("history", []).append(point)
        probe["history"] = probe["history"][-HISTORY_CAP:]
    h = probe.get("history", [])
    first, latest = (h[0] if h else None), (h[-1] if h else None)
    trend = None
    if len(h) >= 4:
        older = sum(x["derive_rate"] for x in h[:len(h) // 2]) / (len(h) // 2)
        newer = sum(x["derive_rate"] for x in h[len(h) // 2:]) / (len(h) - len(h) // 2)
        trend = ("improving" if newer > older + 0.02 else
                 "declining" if newer < older - 0.02 else "flat")
    return {**probe, "status": "measured" if latest else "frozen",
            "frozen_at": probe.get("frozen_cycle"),
            "problem_count": len(probe.get("problems", [])),
            "first_derive_rate": first["derive_rate"] if first else None,
            "latest_derive_rate": latest["derive_rate"] if latest else None,
            "latest_lift": latest["lift"] if latest else None,
            "latest_by_level": latest["by_level"] if latest else None,
            "before_after_gain": (round(latest["derive_rate"] - first["derive_rate"], 3)
                                  if first and latest and len(h) >= 2 else None),
            "trend": trend}
