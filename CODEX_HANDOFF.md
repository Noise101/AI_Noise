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

The latest live v18 run generated the query `きつね つる`, selected `イソップ童話集/きつねとつる`, and induced many repeated chunks. Only `きつね` and `つる` were corroborated by both Japanese Wiktionary and an exact/redirected Japanese Wikipedia page. `つる` remains meaning-ambiguous even though its boundary is accepted.

## Verification

From `experiments/`:

```bash
python3 -m unittest discover -v
```

The complete suite currently has 48 tests and takes about 90 seconds on the development machine.

Live read-only checks:

```bash
python3 developmental_language_v15.py "fox grapes" --output report-v15.json
python3 lexical_research_v16.py "fox grapes" --output report-v16.json
python3 phrase_learning_v17.py "fox grapes" --max-phrases 4 --output report-v17.json
python3 japanese_boundaries_v18.py "きつね つる" --candidate-limit 15 --output report-v18.json
```

## Next concrete work

The next version should turn the separate v12-v19 stages into one persistent autonomous controller. It should choose among causal, concept, word, phrase, boundary, and sense gaps by expected information gain, obey explicit request and test budgets, resume from its evidence ledger, and prove the full loop on unseen seeds.

After that, run an evaluation matrix with previously unseen English and Japanese seed concepts. The required end-to-end gates remain: autonomous gap detection, generated query, read-only retrieval, multiple-source evaluation, concept/causal update, cited conclusion, and demonstrated self-correction.

## Safety and integrity

- Web access is read-only. Do not post, purchase, change permissions, or mutate external services.
- Do not add API keys, deploy keys, credentials, or local machine paths to the repository.
- Do not treat substring search results as lexical validation; v18 requires exact pages or formal redirects.
- Distinguish observed form, candidate boundary, grounded meaning, and causal explanation. Evidence at one level does not prove the next.
- Preserve failed experiments and negative results when they explain a design change.
- Read `RESOURCE_POLICY.md` before running evaluations. Use the quick test profile during iteration, cached web reads, summary output, and the optional 4B Ollama helper only for unverified proposals.
