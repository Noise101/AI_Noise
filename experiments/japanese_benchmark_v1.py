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

BENCH_VERSION = 1
FINAL_QUERY_BUDGET = 2
MIN_SELECTION_STORIES = 8
MIN_FINAL_STORIES = 8
MIN_TRAIN_STORIES = 20
SIGNIFICANCE_Z = 3.0
SELECTION_GROWTH_FOR_FINAL = 1.4     # selection must have stayed significant across
                                    # this much train growth before a final may open


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
                 min_selection: int = MIN_SELECTION_STORIES, min_final: int = MIN_FINAL_STORIES):
        previous = previous or {}
        self.salt = salt
        by_col: dict[str, list] = {}
        for s in stories:
            if len(s.get("events", [])) >= 3:
                by_col.setdefault(collection(s["url"]), []).append(
                    {"url": s["url"], "events": s["events"]})

        # a collection's tier, once assigned, is STICKY (persisted) -- so a
        # collection can never migrate from train into a held-out tier after the
        # model has already trained on it.
        sticky = dict(previous.get("tier_assignments") or {})
        tier_cols: dict[str, set] = {"train": set(), "selection": set(),
                                     "final": set(), "reserve": set()}
        for col in by_col:
            tier_cols[sticky.get(col) or _tier_of(col, salt)].add(col)

        # story-count top-up: the hash split is by collection count, but tiers
        # need a minimum number of STORIES.  When the corpus is big enough overall
        # but a held-out tier came up short, move the lowest-rank `train`
        # collections into it, provided training keeps its own minimum.
        def _stories_in(cols):
            return sum(len(by_col[c]) for c in cols)

        train_ranked = sorted(tier_cols["train"], key=lambda c: _tier_rank(c, salt))
        for tier, need in (("selection", min_selection), ("final", min_final),
                           ("reserve", min_final)):
            i = 0
            while (_stories_in(tier_cols[tier]) < need and i < len(train_ranked)
                   and _stories_in(tier_cols["train"]) - len(by_col[train_ranked[i]]) >= MIN_TRAIN_STORIES):
                c = train_ranked[i]; i += 1
                if c in tier_cols["train"]:
                    tier_cols["train"].discard(c)
                    tier_cols[tier].add(c)
                    sticky[c] = tier
            train_ranked = [c for c in train_ranked if c in tier_cols["train"]]
        self.tier_assignments = {c: t for t in ("selection", "final", "reserve")
                                 for c in tier_cols[t]}

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

        self.selection_frozen = len(self.selection_snapshot) >= min_selection
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

    def report_fields(self) -> dict:
        return {
            "bench_version": BENCH_VERSION,
            "tier_assignments": self.tier_assignments,
            "selection_stories": len(self.selection_snapshot),
            "selection_fingerprint": self.selection_fingerprint,
            "selection_migrated_from_test_snapshot": self.selection_migrated,
            "reserve_stories": len(self.reserve_snapshot),
            "train_stories": len(self.train_stories),
            "tier_collection_counts": self.tier_collection_counts,
            "collection_disjointness": self.disjointness,
            "collections_disjoint": self.disjoint,
        }
