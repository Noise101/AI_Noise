# Codex handoff

## Objective

AI_Noise is an experiment in building a small autonomous learner rather than imitating a large language model. The active direction is:

1. detect its own information gaps;
2. generate a research question and query;
3. browse read-only public sources;
4. keep evidence, citations, source quality, and uncertainty;
5. update character, word, phrase, event, and causal knowledge in parallel;
6. revise beliefs when counterevidence arrives;
7. validate the whole loop on unknown tasks without fixed answer pages.

Do not claim general intelligence or full natural-language understanding. Every version intentionally exposes its limitations.

Read `ARCHITECTURE.md` before changing the learning path. Its invariants are the project constitution: the epistemic core must work without an LLM, and a local-model proposal has no evidential weight.

## Current state

- `v9`: integrated binary causal/concept learner.
- `v10`: probabilistic causal beliefs and calibration.
- `v11`: first web graph attempt; retained as a documented failure because it overuses a ready-made knowledge graph.
- `v12`: child-level event prediction, surprise, Why questions, and falsifiable investigation plans.
- `v13`: read-only public-domain story curriculum using Wikisource and Project Gutenberg.
- `v14`: cross-source concepts, viewpoint separation, citations, disagreement, and belief revision.
- `v15`: parallel character, word-form, phrase-candidate, semantic-role, event, and concept learning.
- `v16`: unknown-word lookup using two Wiktionary projects plus observed story usage; sourced senses are written back.
- `v17`: phrase research; a repeated phrase is not called an idiom until component meanings are grounded.
- `v18`: Japanese word-boundary induction without a pretrained tokenizer, validated against exact Wiktionary/Wikipedia pages.
- `v19`: ambiguous Japanese senses are enumerated from references, grounded in observable story features, cited, and revisable by counter-context.
- `v20`: a persistent budgeted controller selects gaps by expected information gain, saves every cycle, restores learned beliefs, and stops cleanly at network/time/step boundaries.
- `v21-v22`: routine cycles run in a local background worker with a compact heartbeat, resumable per-seed state, evidence-derived curriculum transitions, a safe stop file, and zero Codex/remote-LLM calls. Kanjipedia exact-entry existence is an additional structural reference; its definition prose is not copied.
- `v23-v25`: persistent curiosity grows across repeated unresolved encounters; mastery self-assessment targets the weakest measured language dimension; one bounded local-Ollama conversation per new curriculum supplies practice observations but always has evidence score zero.
- `v26`: global curiosity is referenced through one compact prior file rather than copied into every seed. A tested compactor preserves local cycles/evidence, and the worker enforces a default 1 GB runtime budget every 100 rounds.
- `v27`: one canonical global language memory merges vocabulary, accepted lexical/phrase/dialogue beliefs, event transitions, and concepts. It is loaded before each new story; seed reports retain only local deltas, and global mastery is no longer reset per seed.
- `v28`: deterministic holdout evaluation registers event predictions before comparison and requires independent contexts plus a conservative confidence bound. The live corpus produced zero supported candidates (accuracy 0.0959, equal to baseline), correctly blocking a causal claim until event extraction improves.
- `v29`: transparent event extraction records acceptance/rejection reasons, normalizes auxiliary constructions, and rejects metadata or unresolved pronoun subjects. Legacy events remain available as language history but are quarantined from causal evidence; only audited contiguous events enter `quality_event_transitions`.
- `v30`: removes fox/grapes-specific concept extraction, learns relation-group candidates from repeated distributions, resolves only local explicit coreference, filters and caps the developmental frontier, routes observed Japanese chunks into the Japanese learner, measures Noise's own follow-up skill without crediting partner claims, bases mastery on audited evidence, tests intervention machinery in a separate zero-world-credit lab, and compacts mastery history.

The latest live v18 run generated the query `きつね つる`, selected `イソップ童話集/きつねとつる`, and induced many repeated chunks. Only `きつね` and `つる` were corroborated by both Japanese Wiktionary and an exact/redirected Japanese Wikipedia page. `つる` remains meaning-ambiguous even though its boundary is accepted.

## Verification

From `experiments/`:

```bash
python3 -m unittest discover -v
```

Use `python3 run_tests.py --profile quick --quiet` while iterating, then the full profile before a milestone commit.

Live read-only checks:

```bash
python3 developmental_language_v15.py "fox grapes" --output report-v15.json
python3 lexical_research_v16.py "fox grapes" --output report-v16.json
python3 phrase_learning_v17.py "fox grapes" --max-phrases 4 --output report-v17.json
python3 japanese_boundaries_v18.py "きつね つる" --candidate-limit 15 --output report-v18.json
python3 autonomous_controller_v20.py "fox grapes" --state controller-state.json --max-steps 3 --max-network 8 --summary
python3 local_worker_v21.py start "fox grapes"
python3 local_worker_v21.py status
```

## Next concrete work

Replace the shallow event parser with a generic, transparent narrative-event representation that rejects metadata, normalizes auxiliaries, and keeps provenance. Evaluate it on unseen child-level English and Japanese sentences before rerunning v28. Do not loosen v28 thresholds merely to obtain a positive causal result.

## Safety and integrity

- Web access is read-only. Do not post, purchase, change permissions, or mutate external services.
- Do not add API keys, deploy keys, credentials, or local machine paths to the repository.
- Do not treat substring search results as lexical validation; v18 requires exact pages or formal redirects.
- Distinguish observed form, candidate boundary, grounded meaning, and causal explanation. Evidence at one level does not prove the next.
- Preserve failed experiments and negative results when they explain a design change.
- Read `RESOURCE_POLICY.md` before running evaluations. Use the quick test profile during iteration, cached web reads, summary output, and the optional 4B Ollama helper only for unverified proposals.
