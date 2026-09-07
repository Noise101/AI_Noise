#!/usr/bin/env python3
"""Tiered frozen-benchmark bookkeeping for the Japanese capability tests.

The re-audit (#4) found that "measure the same frozen snapshot twice at two
training sizes = confirmed" turns that snapshot into a development set: you keep
looking at the answer.  This module splits every Japanese capability test into

  * a SELECTION snapshot -- frozen once, measured as often as you like for
    learning-curve / model-comparison / bottleneck diagnosis.  Beating the
    baseline here is NEVER a capability claim; the measurement count is recorded.

  * a FINAL snapshot -- collection-disjoint from BOTH training and selection,
    opened at most `FINAL_QUERY_BUDGET` times, each open on a snapshot that was
    never opened before.  The model, regime, scoring version, baseline and
    threshold are frozen from the selection side BEFORE the final is scored.
    Capability = selection-significant AND an unopened-final measurement that
    also clears the pre-registered threshold.

  * a RESERVE snapshot -- a second independent final, held for a retry after a
    final failure (or a model change that invalidates the recorded final).

Collections (a work, its sub-pages, an Aozora file series) never straddle
train / selection / final / reserve.  Bucketing is a stable hash of the
collection key, so a source keeps its tier as the corpus grows; snapshots are
frozen the first time a tier is large enough and never re-drawn.
"""

from __future__ import annotations

import hashlib
import json

BENCH_VERSION = 2
FINAL_QUERY_BUDGET = 2
MIN_SELECTION_STORIES = 8
MIN_FINAL_STORIES = 8
MIN_TRAIN_STORIES = 20
SIGNIFICANCE_Z = 3.0
STREAK_GROWTH = 1.4          # train must grow this much between selection-streak steps
CANDIDATE_GROWTH = 1.5       # ... and this much beyond the last candidate anchor to
                            # register a NEW candidate checkpoint (opens a reserve final)


def advance_selection_streak(prev: dict | None, significant: bool,
                             train_n: int, model_fp: str) -> dict:
    """Anchor-based selection streak (re-audit #6 P1-3).

    The streak counts *independent* significant measurements: the first fixes an
    anchor (train size + model fingerprint) and gives streak 1; each further step
    needs the training set to have grown STREAK_GROWTH x since the anchor AND a
    different model.  The anchor is NOT re-stamped otherwise, so ordinary gradual
    growth (160->180->...->430) eventually reaches streak 2.  A non-significant
    measurement resets streak and anchor to nothing.
    """
    if not significant:
        return {"streak": 0, "anchor_train": None, "anchor_model": None,
                "next_streak_train": None}
    p = prev or {}
    streak, at, am = p.get("streak", 0), p.get("anchor_train"), p.get("anchor_model")
    if streak <= 0 or at is None:
        return {"streak": 1, "anchor_train": train_n, "anchor_model": model_fp,
                "next_streak_train": int(train_n * STREAK_GROWTH) + 1}
    if train_n >= at * STREAK_GROWTH and model_fp != am:
        return {"streak": streak + 1, "anchor_train": train_n, "anchor_model": model_fp,
                "next_streak_train": int(train_n * STREAK_GROWTH) + 1}
    return {"streak": streak, "anchor_train": at, "anchor_model": am,
            "next_streak_train": int(at * STREAK_GROWTH) + 1}


def should_register_candidate(candidates: list, streak: int, train_n: int) -> bool:
    """A candidate checkpoint is registered (and its one-shot final opened) only
    when selection has reached streak 2 and either nothing is registered yet or
    the training set has grown CANDIDATE_GROWTH x beyond the last registered
    candidate.  A one-book / few-step change never registers a new candidate, so
    the reserve is never burned by ordinary continued training (P1-4)."""
    if streak < 2 or len(candidates) >= FINAL_QUERY_BUDGET:
        return False
    if not candidates:
        return True
    return train_n >= candidates[-1].get("registered_at_train", 0) * CANDIDATE_GROWTH


def capability_view(candidates: list, current_model_fp: str) -> dict:
    """`confirmed_ever` = some candidate checkpoint passed its unopened final.
    `confirmed_current_model` = the model that is running RIGHT NOW is itself a
    confirmed checkpoint (rare for a continuously-training model -- shown so the
    two are never conflated)."""
    confirmed = [c for c in candidates if (c.get("final_result") or {}).get("significant")]
    last = confirmed[-1] if confirmed else None
    return {
        "confirmed_ever": bool(confirmed),
        "confirmed_current_model": bool(last and last.get("model_fingerprint") == current_model_fp),
        "confirmed_checkpoints": [
            {"registered_at_train": c.get("registered_at_train"),
             "model_fingerprint": c.get("model_fingerprint"),
             "final_snapshot_fingerprint": c.get("final_snapshot_fingerprint"),
             "final_z": (c.get("final_result") or {}).get("z") or (c.get("final_result") or {}).get("gain_z"),
             "confirmed_at": c.get("confirmed_at")}
            for c in confirmed],
        "last_confirmed_checkpoint": last,
        "candidates_registered": len(candidates),
        "candidate_budget": FINAL_QUERY_BUDGET,
    }


def collection(url: str) -> str:
    """Group a multi-part source so its parts never straddle a split.  A bare
    /wiki/<Title> page (parent is only the generic /wiki mount) is its own
    collection; /wiki/<Collection>/<Title> and /cards/NNN/files/xxx group on the
    parent."""
    base = url.split("#")[0].split("?")[0].rstrip("/")
    parts = base.split("/")
    if len(parts[3:]) >= 3:
        return "/".join(parts[:-1])
    return base


def _tier_rank(col: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}|tier|{col}".encode()).hexdigest(), 16) % 100


def _tier_of(col: str, salt: str) -> str:
    h = _tier_rank(col, salt)
    if h < 12:
        return "final"
    if h < 18:
        return "reserve"
    if h < 42:
        return "selection"
    return "train"


def _normalise_ledger(raw) -> dict:
    """Accept the old {col: 'tier'} form and the new {col: {...}} form."""
    out: dict[str, dict] = {}
    for col, v in (raw or {}).items():
        if isinstance(v, str):
            out[col] = {"tier": v, "assignment_reason": "legacy", "assigned_at": 0,
                        "ever_trained": v == "train"}
        elif isinstance(v, dict):
            out[col] = {"tier": v.get("tier", "train"),
                        "assignment_reason": v.get("assignment_reason", "legacy"),
                        "assigned_at": v.get("assigned_at", 0),
                        "ever_trained": bool(v.get("ever_trained"))}
    return out


def canonical_events(events: list) -> list:
    out = []
    for e in events:
        if isinstance(e, dict):
            out.append({"subject": e.get("subject", ""), "verb": e.get("verb", ""),
                        "obj": e.get("obj", "")})
        else:
            out.append({"subject": e[0] if len(e) > 0 else "",
                        "verb": e[1] if len(e) > 1 else "",
                        "obj": e[2] if len(e) > 2 else ""})
    return out


def fingerprint(snapshot: list) -> str:
    """Hash URL *and* canonical event content of every snapshot story."""
    payload = json.dumps(
        sorted(({"url": s["url"], "events": canonical_events(s["events"])} for s in snapshot),
               key=lambda s: s["url"]),
        sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Tiers:
    """The train / selection / final / reserve view of the read stories for one
    benchmark, honouring any snapshot already frozen in `previous`."""

    def __init__(self, stories: list, salt: str, previous: dict | None = None,
                 min_selection: int = MIN_SELECTION_STORIES, min_final: int = MIN_FINAL_STORIES,
                 ever_trained_collections: "set | None" = None, cycle: int = 0):
        previous = previous or {}
        self.salt = salt
        ever_trained = set(ever_trained_collections or ())
        by_col: dict[str, list] = {}
        for s in stories:
            if len(s.get("events", [])) >= 3:
                by_col.setdefault(collection(s["url"]), []).append(
                    {"url": s["url"], "events": s["events"]})

        # --- FULL, persisted tier ledger (re-audit #6 P1-2) ---
        # {col: {tier, assigned_at, assignment_reason, ever_trained}}.  EVERY
        # collection is recorded, train included.  A collection that has ever been
        # trained on -- or was previously assigned `train` -- is pinned to train
        # and can NEVER move to a held-out tier.  Top-up only ever touches
        # collections that are fresh (no prior assignment, never trained).
        prev_ledger = _normalise_ledger(previous.get("tier_assignments"))
        ledger: dict[str, dict] = {}
        tier_cols: dict[str, set] = {"train": set(), "selection": set(),
                                     "final": set(), "reserve": set()}
        for col in by_col:
            prev = prev_ledger.get(col)
            is_trained = col in ever_trained or (prev and prev.get("tier") == "train") \
                or (prev and prev.get("ever_trained"))
            if is_trained:
                tier, reason = "train", ("ever_trained" if col in ever_trained else
                                         (prev or {}).get("assignment_reason", "train_pinned"))
            elif prev:
                tier, reason = prev["tier"], prev.get("assignment_reason", "sticky")
            else:
                tier, reason = _tier_of(col, salt), "hash"
            tier_cols[tier].add(col)
            ledger[col] = {"tier": tier, "assignment_reason": reason,
                           "assigned_at": (prev or {}).get("assigned_at", cycle),
                           "ever_trained": bool(is_trained)}

        # story-count top-up: a held-out tier below its story minimum pulls the
        # lowest-rank FRESH (never-assigned, never-trained) train collections in.
        def _stories_in(cols):
            return sum(len(by_col[c]) for c in cols)

        fresh_train = sorted((c for c in tier_cols["train"]
                              if not ledger[c]["ever_trained"] and c not in prev_ledger),
                             key=lambda c: _tier_rank(c, salt))
        self.topup_short = {}
        for tier, need in (("selection", min_selection), ("final", min_final),
                           ("reserve", min_final)):
            for c in list(fresh_train):
                if _stories_in(tier_cols[tier]) >= need:
                    break
                if _stories_in(tier_cols["train"]) - len(by_col[c]) < MIN_TRAIN_STORIES:
                    break
                tier_cols["train"].discard(c)
                tier_cols[tier].add(c)
                fresh_train.remove(c)
                ledger[c] = {"tier": tier, "assignment_reason": "topup_fresh_untrained",
                             "assigned_at": cycle, "ever_trained": False}
            if _stories_in(tier_cols[tier]) < need:
                self.topup_short[tier] = _stories_in(tier_cols[tier])
        self.tier_assignments = ledger

        # frozen snapshots -- once captured, reused verbatim
        self.selection_snapshot = list(previous.get("selection_snapshot") or [])
        self.reserve_snapshot = list(previous.get("reserve_snapshot") or [])
        # finals already opened: [{tier, fingerprint, preconditions:{final_snapshot_urls}}]
        self.final_history = list(previous.get("final_history") or [])
        self._used_final_tiers = {f.get("tier") for f in self.final_history}
        self.selection_migrated = False

        # one-time migration: a pre-tier `test_snapshot` becomes the selection set
        if not self.selection_snapshot and previous.get("test_snapshot"):
            self.selection_snapshot = [
                {"url": s["url"], "events": s["events"]} for s in previous["test_snapshot"]]
            self.selection_migrated = True

        if not self.selection_snapshot:
            cand = [s for c in sorted(tier_cols["selection"]) for s in by_col.get(c, [])]
            if len(cand) >= min_selection:
                self.selection_snapshot = cand

        if not self.reserve_snapshot:
            cand = [s for c in sorted(tier_cols["reserve"]) for s in by_col.get(c, [])]
            if len(cand) >= min_final:
                self.reserve_snapshot = cand

        # candidate primary final: its bucket must be big enough and (guaranteed
        # here) never trained on -- `train` below excludes every non-train tier
        if "final" not in self._used_final_tiers:
            cand = [s for c in sorted(tier_cols["final"]) for s in by_col.get(c, [])]
            self._pending_primary_final = cand if len(cand) >= min_final else []
        else:
            self._pending_primary_final = []
        self._primary_final_ready = bool(self._pending_primary_final)

        self.min_selection, self.min_final = min_selection, min_final
        self.selection_frozen = len(self.selection_snapshot) >= min_selection
        self.selection_insufficient_reason = (None if self.selection_frozen else
            ("fresh_untrained_collections_short_for_selection"
             if "selection" in self.topup_short else "not_enough_stories_yet"))
        sel_cols = {collection(s["url"]) for s in self.selection_snapshot}
        used_final_cols = {collection(u) for f in self.final_history
                           for u in f.get("preconditions", {}).get("final_snapshot_urls", [])}
        reserve_cols = {collection(s["url"]) for s in self.reserve_snapshot}
        pending_final_cols = {collection(s["url"]) for s in self._pending_primary_final}
        forbidden = (tier_cols["selection"] | tier_cols["final"] | tier_cols["reserve"]
                     | sel_cols | used_final_cols | reserve_cols | pending_final_cols)
        self.forbidden_train_collections = forbidden

        self.train_stories = [s for c in sorted(tier_cols["train"]) for s in by_col.get(c, [])
                              if collection(s["url"]) not in forbidden]
        self.tier_collection_counts = {k: len(v) for k, v in tier_cols.items()}
        self._by_col = by_col

        train_cols = {collection(s["url"]) for s in self.train_stories}
        all_final_cols = used_final_cols | pending_final_cols
        self.disjointness = {
            "train_x_selection": sorted(train_cols & sel_cols),
            "train_x_final": sorted(train_cols & all_final_cols),
            "selection_x_final": sorted(sel_cols & all_final_cols),
            "train_x_reserve": sorted(train_cols & reserve_cols),
        }
        self.disjoint = not any(self.disjointness.values())
        self.selection_fingerprint = fingerprint(self.selection_snapshot) if self.selection_snapshot else None

    # --- final management --------------------------------------------------
    def next_unopened_final(self) -> "dict | None":
        """The next final snapshot to open: the pending primary, else the reserve
        (once the primary was opened), else None."""
        if self._primary_final_ready and "final" not in self._used_final_tiers:
            snap = self._pending_primary_final
        elif "final" in self._used_final_tiers and "reserve" not in self._used_final_tiers \
                and len(self.reserve_snapshot) >= MIN_FINAL_STORIES:
            snap = self.reserve_snapshot
        else:
            return None
        tier = "final" if snap is self._pending_primary_final else "reserve"
        return {"tier": tier, "urls": [s["url"] for s in snap], "snapshot": snap,
                "fingerprint": fingerprint(snap)}

    def record_final(self, opened: dict) -> None:
        self._used_final_tiers = self._used_final_tiers | {opened["tier"]}

    def next_final_ready(self) -> bool:
        return self.next_unopened_final() is not None

    def final_insufficient_reason(self) -> "str | None":
        if self.next_unopened_final() is not None:
            return None
        if "final" in self.topup_short and "final" not in self._used_final_tiers:
            return "fresh_untrained_collections_short_for_final"
        if "reserve" in self.topup_short and "reserve" not in self._used_final_tiers:
            return "fresh_untrained_collections_short_for_reserve"
        return "all_final_and_reserve_snapshots_used"

    def report_fields(self) -> dict:
        trained = sum(1 for v in self.tier_assignments.values() if v.get("ever_trained"))
        return {
            "bench_version": BENCH_VERSION,
            "tier_assignments": self.tier_assignments,
            "tier_collections_ever_trained": trained,
            "topup_short_tiers": self.topup_short,
            "selection_stories": len(self.selection_snapshot),
            "selection_fingerprint": self.selection_fingerprint,
            "selection_frozen": self.selection_frozen,
            "selection_insufficient_reason": self.selection_insufficient_reason,
            "selection_migrated_from_test_snapshot": self.selection_migrated,
            "reserve_stories": len(self.reserve_snapshot),
            "train_stories": len(self.train_stories),
            "tier_collection_counts": self.tier_collection_counts,
            "collection_disjointness": self.disjointness,
            "collections_disjoint": self.disjoint,
        }
