#!/usr/bin/env python3
"""Curriculum candidate re-weighting and eligibility, kept separate from
candidate *generation*.

Seven functions in local_worker_v21.py each hard-code a base "score" literal
when they build a candidate dict: discover_curriculum() (three literals plus
one Japanese-boundary case), discover_from_developmental_shelves() (a decaying
shelf_score), and five single-candidate "counterexample_hunt" helpers
(parser_counterexample_candidate, structural_counterexample_candidate,
learned_rule_boundary_candidate, causal_comparison_candidate,
repeated_grounding_candidate). This module does not touch any of those seven
call sites or centralize their literals into a lookup table -- each score is
tied to reasoning specific to how that candidate was found, and folding them
into one constants table was judged not worth the churn (see the audit thread
that requested this file). What *is* centralized here is the two functions
that consume those base scores afterward:

- learned_curriculum_score() re-weights whichever base score a candidate
  arrives with by how well that candidate's `reason` has actually paid off
  historically (a Beta-smoothed admission rate from strategy_performance).
- curriculum_strategy_allowed() filters out a reason entirely once
  update_curriculum_strategy (still in local_worker_v21.py -- it records
  outcomes, this module only reads them) has marked it "deprioritized".

Both have exactly one production call site, together, in local_worker_v21's
work() when the frontier is rebuilt.

Reference: what each generator's hard-coded base score means today
-------------------------------------------------------------------
discover_curriculum():
  3.0  "unvisited story link found in read evidence" -- an explicit, observed
       hyperlink between two already-read pages; the strongest continuity
       signal available
  2.5  "unvisited page in an observed story collection" -- same shelf as an
       admitted page, but not a link actually followed in the text
  1.5  "repeated unsegmented Japanese chunks require boundary grounding"
  0.5  "unvisited concept pair from evidence ledger" -- lowest: an inferred
       pairing, not an observed reading path

discover_from_developmental_shelves():
  shelf_score starts from a per-shelf prior (DEVELOPMENTAL_SHELVES) and decays
  by 0.25 per subcategory depth crossed, floored at 0.5 -- a title several
  categories deep into an unrelated shelf is a weaker signal than one on a
  top-level shelf.

Single-candidate counterexample_hunt helpers (each returns at most one
candidate, so their relative ranking only matters when more than one could
fire in the same cycle):
  7.0  learned_rule_boundary_candidate() -- probes a learned rule's own
       documented boundary case: the most targeted, highest-effort probe
  6.5  causal_comparison_candidate()
  6.25 repeated_grounding_candidate()
  6.0  structural_counterexample_candidate()
  5.5  parser_counterexample_candidate() -- lowest of the five: a parser
       failure alone is the weakest signal that a targeted seed will actually
       produce useful evidence

All five sit comfortably above discover_curriculum()'s exploratory range
(0.5-3.0): a targeted counterexample search during counterexample_hunt should
usually outrank ordinary exploration once one is available.
"""

from __future__ import annotations


def curriculum_strategy_allowed(curriculum: dict, candidate: dict) -> bool:
    performance = curriculum.get("strategy_performance", {}).get(candidate.get("reason"), {})
    return performance.get("status") != "deprioritized"


def learned_curriculum_score(curriculum: dict, candidate: dict) -> float:
    """Rank routes by their observed developmental yield with a Beta prior."""
    base = candidate.get("score", 0.0)
    performance = curriculum.get("strategy_performance", {}).get(candidate.get("reason"), {})
    admitted, rejected = performance.get("admitted", 0), performance.get("rejected", 0)
    expected_yield = (admitted + 1) / (admitted + rejected + 2)
    return base * (0.5 + expected_yield)
