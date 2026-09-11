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
- `v31`: compares surface, role, role+object-head, and experience-derived spelling-family event representations on deterministic holdout data. An abstraction is adopted and revision-logged only if it beats both surface and frequency baseline. The first live evaluation increased coverage to 0.413 but tied baseline accuracy at 0.0435, so it correctly retained surface representation. Grounded component meanings may create revisable phrase compositions; repeated dialogue punctuation patterns may ground structural conversation functions.
- `v32`: scores each audited source by narrative-event yield, child-length sentences, words recurring across at least three curricula, subject recurrence, and dialogue form. Only current-level sources enter canonical memory or spawn descendants. Migration archives pre-v32 memory/curriculum and rebuilds both from raw reports; a preflight admitted 11 of 939 reports and rejected biography, encyclopedia, and travel-text drift.
- `v33-v36`: adds association learning, an observation-only human-science scaffold, persistent error memory, and image-depiction memory. Association is explicitly not causal evidence and image depictions are not treated as real-world perception.
- `v37-v40`: retains failed structural rules and parser counterexamples, compares parser policies on held-out sources, audits every accepted/rejected sentence, and quarantines legacy parsing errors.
- `v41-v46`: tests active causality, tools, other agents, cooperation, and abstraction in bounded procedural worlds. These stages are mechanism diagnostics only and contribute zero credit to real-reading capability.
- `v47-v49`: rebuilds prediction evidence from retained source sentences, uses whole-source holdouts, and selects developmental sentence complexity only on unseen-source performance.
- `v48`: local-model conversation produces hypotheses with zero evidence credit; only independently observed Web contexts may support a structural hypothesis.
- `v50`: structures independently sourced experiences, retains counterexamples, and promotes a rule only after source-held-out success.
- `v51`: introduced state/goal/action/result observation frames and a locked narrative benchmark, but every next-event predictor (world_model_v51, association_learning_v33, causal_experiment_v28, representation_learning_v31) stayed at the frequency baseline. **All four were retired** and replaced by `event_structure_v1`, which predicts structure *inside* one event (verb_cloze, event_plausibility). See `ARCHITECTURE.md` "Predict within the event, not the next event".

The active real-material path is `local_worker_v21.py` plus `event_structure_v1.py`.
`world_model_v51.py` and the other next-event predictors are DELETED -- do not
resurrect them.  `unified_agent_v9.py` is only the bounded binary-world experiment.
A parallel developmental Japanese reading loop
(`japanese_reader_v1.py` + `japanese_event_v1.py` + `reading_curriculum_v1.py`) runs
inside the same worker; its state lives in `.local/reading-*.json`.

The latest live v18 run generated the query `きつね つる`, selected `イソップ童話集/きつねとつる`, and induced many repeated chunks. Only `きつね` and `つる` were corroborated by both Japanese Wiktionary and an exact/redirected Japanese Wikipedia page. `つる` remains meaning-ambiguous even though its boundary is accepted.

## Verification

From `experiments/`:

```bash
python3 -m unittest discover -v
```

Use `python3 run_tests.py --profile quick --quiet` while iterating, then the full profile before a milestone commit.

External auditors that cannot browse GitHub trees can use these immutable raw URLs:

- https://raw.githubusercontent.com/Noise101/AI_Noise/main/experiments/event_structure_v1.py
- https://raw.githubusercontent.com/Noise101/AI_Noise/main/experiments/local_worker_v21.py
- https://raw.githubusercontent.com/Noise101/AI_Noise/main/experiments/test_event_structure_v1.py
- https://raw.githubusercontent.com/Noise101/AI_Noise/main/experiments/test_local_worker_v21.py

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

## Japanese reading loop — evaluation integrity (2026-09-10)

Three evaluation holes in the Japanese loop were closed. Each keeps the old
state under `.local/audit/` and does not inherit its passes:

- **P1-1 — retelling had no valid likelihood model.** The order-recovery
  benchmark scored the *general* character RNN, whose corpus is the raw text of
  every read book (Tatoeba bundles + unparseable books included) — a permanent
  superset of the narrative story set — so `baseline_corpus_matches_rnn` was
  always False (`measurement_invalid_baseline_corpus_mismatch`). Now a
  **dedicated narrative RNN** (`japanese_sequence_v1`, regime `jnarr_seq_v1`,
  `.local/reading-narrative-sequence.json`) trains only on recognised-narrative
  raw text minus every held-out collection; the benchmark uses it and its
  training URLs are the position baseline's source set by construction. The
  general model is kept, unchanged, as the diagnostic / display model.
- **P1-2 — dialogue scored echoes as understood.** `japanese_dialogue_v1.score`
  now removes everything the reply copied from Noise's own sentence, requires
  independent semantic content of the *kind* the utterance form asks for, and
  records `echo_detected / clarification / relevant_new_information /
  response_to_requested_act / contradicts_belief / partner_guessed_malformed`
  plus utterance quality (`parseable / belief_supported / relation_supported /
  malformed`). The probe freezes the *task* (concept + fixed intent), not just
  the concept. VERSION 3.
- **P1-3 — capability probe conditioned its baseline to fail.** `build_probe`
  kept only baseline-fails problems, so `lift` was the raw derive-rate.
  `capability_probe_v1` VERSION 3 splits into **challenge** (baseline-fails,
  diagnostic only), **unbiased selection** (salt sample, no gold/baseline
  filter), and **unopened final + reserve** (graded once, only after a
  pre-registered selection threshold clears on two distinct model
  fingerprints). Each word is hashed to a tier before any gold is seen; a
  problem lives in the tier of the word it asks about, so no concept straddles a
  split (a held-out problem's distractors may be challenge-tier or same-tier,
  never a different held-out tier). `selection_refs` (the reference pool) is
  decoupled from `TEST_SET_TARGET` — it keeps growing to `REF_POOL_TARGET` so
  the probe can reach its tiers even after the frozen capability set is full.

`capability_confirmed` on the probe now needs ALL of: unbiased selection cleared
the pre-registered bar on two distinct models; an unopened final improved in the
same direction; the final beat its baseline by the minimum effect; tier
separation valid. Retelling `capability_confirmed` still needs a candidate
checkpoint passing its one-shot unopened final (unchanged).

## Next concrete work

Improve the v51 heuristic observation parser without changing the locked benchmark examples. The present state/goal frames are auditable but still limited to explicit simple clauses. A new parser or representation must beat the frequency baseline on both the source-disjoint selection set and untouched final set; do not loosen thresholds to manufacture a positive result.

The Japanese-side rate limiter is `japanese_word_meaning_v1`'s held-out concrete
vocabulary (~19 words with a wiktionary-validatable genus). It caps the cognition
probe's tiers AND the dialogue probe. Faster concrete-noun throughput is the
highest-leverage next step; do not lower `MIN_SELECTION_PROBLEMS` /
`TEST_SET_TARGET` to manufacture a signal.

## The predict → fail → self-correct loop (`japanese_prediction_v1`, 2026-09-10)

Added the step the Japanese loop was missing: Noise scores a real story event's
plausibility from its concepts, scores a genus-corrupted version, and a real
event it keeps rating implausible feeds `japanese_word_meaning_v1` a capped
(≤0.30) counter-evidence against the subject's believed genus (`_revise_belief`,
one channel). Tiered like the other benchmarks; secondary diagnostic = next-verb
-class prediction.

**Result on the live corpus: near-flat, and it localises the bottleneck.**
Next-verb-class prediction is ~+0.3pt over the marginal (dead, same as English
`verb_cloze`). Plausibility discrimination is ~+0.6pt full / ~+1.6pt on the
~22% of trials that carry discriminative structure — not significant — because
**90% of scorable events are `(creature subject, no object genus)`**. The
concept graph is ~78% 生き物; the concrete non-creature nouns (道具/場所/食べ物 as
narrative arguments) that would make events checkable are the same
held-out-scarce set. Every measured Japanese-side capability (word_meaning z,
cognition probe, retell, dialogue, and now prediction) bottlenecks on the same
thing: **the concrete vocabulary Noise can *validate a genus for* from this
corpus.**

**2026-09-10 — that bottleneck was substantially a bug, not corpus poverty.**
`japanese_word_meaning_v1` had researched ~369 words but `_coarse` (fine genus →
coarse fold) placed only ~102, and `selection_refs` cached *transient fetch
failures* permanently (267 of 348 held-out probes had an empty genus, never
retried — 太陽 / ランプ / お金 / 手紙 / 人形 / 通り). Fixes:

- A **10th coarse class 身体** (body / organ / body-part / bodily fluid) added to
  `COARSE_CLASSES`, mirrored in `cognition_v1.COARSE`,
  `japanese_dialogue_v1.COARSE`, `japanese_prediction_v1._GENUS_VC_FIT`.
- `_coarse` reworked to a **two-pass matcher**: pass 1 needs the class key to be
  a SUFFIX of the genus string (the definition head noun — 鍵盤楽器→道具,
  教育施設→場所, 内臓の一つ→身体), which structurally removes most mid-word
  collisions (木管楽器 no longer 植物 via 木; 水族館 no longer 自然物 via 水; 神社→場所);
  pass 2 is the old anywhere-substring with a small `_COARSE_BLOCK` veto
  (末梢神経障害 contains 神). Rule order (生き物→人→身体→植物→…) breaks ties.
- `_COARSE_RULES` keyword coverage widened across every class.
- `_wiktionary_gist` extraction widened (more `…である`/…をいう/…と呼ばれ patterns,
  last-run fallback, wikipedia `exsentences` 1→2).
- `selection_refs` empty-genus entries **retried up to `REFS_MAX_ATTEMPTS`**.

Offline: `_coarse`-placeable researched words 102 → 142, 0 regressions. On the
live worker the held-out set moved 20 → 23 in ~10 cycles and keeps climbing.
Feeds ALL five stuck signals. `_coarse` is normalisation of an external source's
wording — a word is still "understood" only on its own reading evidence.
Still TODO if this is not enough: a graded developmental corpus, or parser
object-extraction.

## Parser v9 — structural morphological analysis (2026-09-11)

The pure particle-anchor heuristic could not break down the multi-clause,
relative-clause-heavy sentences that make up most of the corpus (measured
explicit-subject rate ~57% on the `shelved_stuck` shelf). `japanese_event_v1`
now uses the vendored/system morphological analyser (`morphology_teacher.py`)
as **structural parse infrastructure** when it is installed — see
`ARCHITECTURE.md` "Structural morphological analysis vs. semantic proposal" for
the invariant-10 boundary this draws: deterministic segmentation/POS/lemma is
not a claim about meaning, so its output is stamped `heuristic_self` like the
old heuristic parse and may feed learning; interpretation of what the recovered
structure *means* still runs entirely on Noise's own heuristics.

`_morph_sentence_events` recovers case-marked subject/verb/object structure
per sentence and falls back to the particle heuristic when the analysis looks
degenerate — several concrete failure modes of the vendored IPADIC dictionary
were found and gated on the live corpus, not assumed away:

- a long **kana-only** sentence is frequently mis-segmented (「いりくち」→
  いる+くちる) — deferred to the heuristic above a length floor;
- a volitional form (「行こう」) surfaces as a raw, un-conjugated base — repaired
  to the dictionary form;
- a 未然形 token immediately followed by a case particle is almost always a
  hiragana-written **noun** misread as a verb stem (「やま」(山) → 動詞やむ,未然形
  before へ) — rejected as a predicate;
- an **out-of-dictionary proper noun** (a character name) glues into a nonsense
  verb complex (「ももたろうは」→ もも+たろ+う+はお(動詞)) — detected by comparing
  against the raw text's own unambiguous topic/subject marking
  (`_LEADING_TOPIC`) and deferred to the heuristic;
- a relative-clause head, a chain of more than `_MORPH_MAX_CHAIN` predicates, or
  an unstripped clause fragment standing in for a subject are all rejected
  (`_strip_modifier`, `_morph_subject_ok`).

`PARSER_VERSION` 8 → 9. `reevaluate_stale_parses` puts every `shelved_stuck` /
`unparsable` book at/below the level back into rotation and drops its cached
events, so the shelf re-reads under the new parser rather than inheriting old
failures; `floor_level` protects the already-earned level from the parser's
level re-walk. Measured on the 25 fetchable `shelved_stuck` books at the live
level (2026-09-11): heuristic-only 616 events / 354 explicit-subject (57%) vs.
morph-assisted 977 events / 676 explicit-subject (69%) — more events recovered
per book AND a materially higher fraction with a named (not guessed) subject.
`AI_NOISE_NO_MORPHOLOGY=1` still forces the pure-heuristic path; new tests in
`test_japanese_event_v1.MorphologyDrivenParseTest` cover the recovery cases and
every gate above.

Not yet done: object-genus coverage (still the bottleneck noted above) is
unaffected by this change — it improves *structure* recovery, not which nouns
get a validated genus. Whether the extra structural yield actually moves the
frozen retelling/probe/prediction numbers needs several read cycles on the
restarted worker to observe.

## Frozen dialogue probe was contaminated by a vocabulary bug (2026-09-11)

`japanese_dialogue_v1`'s `overall_understood_rate` looked stagnant/negative
(`before_after_gain: -0.166`) with no obvious cause. Root cause: two of its 12
permanently-frozen probe concepts, `も気` and `も返事`, were not real words —
the particle scanner had left a topic particle (も) glued to the front of a
real noun (気/返事), and `japanese_word_meaning_v1._norm`'s glued-particle
stripper only fired when the remainder was a known STOP word (its original
purpose: `と彼` → `かれ`), so these survived as "words" and were reinforced by
`reading_usage` evidence into `understood: True` beliefs. Once frozen into the
probe's concept set (by design, to stop the metric's goalposts moving), no
amount of further reading could ever fix them — they are not real words no
matter how much text is read.

Fixed generally, not by special-casing these two strings: `_norm` now asks the
same morphological analyser the event parser uses whether a `(particle)(word)`
candidate structurally segments as two tokens (particle + noun) rather than
one; if so, the particle is stripped regardless of whether the remainder
happens to be a stop word. Real words that merely start with the same kana as
a particle (とけい, となり) are unaffected — the analyser tokenises them as one
word. `japanese_dialogue_v1.VERSION` 3 → 4 archives the contaminated frozen
set to `.local/audit/` (the same version-bump-retires mechanism used
elsewhere) and re-freezes from the now-clean concept pool; the two corrupted
beliefs were also removed from the live `.local/reading-word-meaning.json` as
a one-time data-hygiene fix (they were never real observations to begin with).

This does not mean every "flat" or "declining" number in this project is a
bug: `cognition_probe`'s `challenge_before_after_gain: -0.096` that surfaced
in the same conversation is ordinary sampling noise on a 42-item, admittedly
non-capability diagnostic tier (its own `note` field says so) that has
fluctuated between 0.095 and 0.227 for 65 measurements running — the tier that
matters, `selection`, is trending `improving`. Read the full history before
calling a small-n diagnostic swing a regression.

## Safety and integrity

- Web access is read-only. Do not post, purchase, change permissions, or mutate external services.
- Do not add API keys, deploy keys, credentials, or local machine paths to the repository.
- Do not treat substring search results as lexical validation; v18 requires exact pages or formal redirects.
- Distinguish observed form, candidate boundary, grounded meaning, and causal explanation. Evidence at one level does not prove the next.
- Preserve failed experiments and negative results when they explain a design change.
- Read `RESOURCE_POLICY.md` before running evaluations. Use the quick test profile during iteration, cached web reads, summary output, and the optional 4B Ollama helper only for unverified proposals.
