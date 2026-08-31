#!/usr/bin/env python3
"""Run a cheap development test profile or the full release profile."""

from __future__ import annotations

import argparse
import sys
import time
import unittest


QUICK_MODULES = [
    "test_architecture_contract",
    "test_autonomous_controller_v20",
    "test_local_worker_v21",
    "test_kanjipedia_reference_v22",
    "test_curiosity_drive_v23",
    "test_mastery_drive_v24",
    "test_local_conversation_v25",
    "test_compact_runtime_v26",
    "test_global_memory_v27",
    "test_causal_experiment_v28",
    "test_narrative_event_v29",
    "test_causal_lab_v30",
    "test_story_learning_v12", "test_story_web_curriculum_v13", "test_story_concepts_v14",
    "test_developmental_language_v15", "test_lexical_research_v16", "test_phrase_learning_v17",
    "test_japanese_boundaries_v18",
    "test_japanese_sense_grounding_v19",
    "test_local_candidate_helper", "test_web_cache",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    loader = unittest.defaultTestLoader
    suite = loader.discover(".") if args.profile == "full" else unittest.TestSuite(
        loader.loadTestsFromName(module) for module in QUICK_MODULES)
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=1 if args.quiet else 2).run(suite)
    print(f"profile={args.profile} tests={result.testsRun} seconds={time.monotonic() - started:.3f}")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
