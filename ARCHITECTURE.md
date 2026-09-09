# Architecture constitution

## Purpose

AI_Noise is not a reduced imitation of a huge language model. It must form concepts and causal relations from limited experience, predict before observing outcomes, and correct itself when predictions fail.

This principle outranks convenience, benchmark appearance, fluent output, and short-term feature count.

## Non-negotiable invariants

1. **The core learner works without an LLM.** Character, boundary, word, phrase, event, concept, causal, uncertainty, and belief-revision paths must remain executable with no pretrained model available.
2. **Observation precedes belief.** A stored belief must trace to an observation, intervention, source, or explicit counterexample. A generated sentence is not an observation.
3. **Prediction precedes correction.** Whenever the environment allows it, the agent records a prediction before seeing the result. Self-correction is measured against that prior prediction, not a retrospective explanation.
4. **Small experience matters.** Evaluations report sample count and compare learning speed, not only final accuracy after large ingestion.
5. **Concepts compress and transfer.** A useful concept should explain multiple observations or reduce relearning cost. Memorized surface sequences are not automatically concepts.
6. **Causality requires contrast.** Temporal order and co-occurrence may create a candidate cause, but causal confidence requires comparison, intervention, or falsification evidence.
7. **Uncertainty is retained.** Missing evidence, polysemy, viewpoint differences, and genuine contradictions stay explicit. The system may answer `unknown`.
8. **Failure changes the learner.** Counterevidence must be able to weaken, replace, split, or retire a belief, with the revision retained in history.
9. **Web content is evidence, not authority.** Multiple sources, provenance, hashes, scope, and source independence are tracked. Read-only browsing never grants truth automatically.
10. **Optional local AI has no vote.** A local model may propose bounded candidates or queries. Its output always begins unverified with evidence score zero and cannot directly update knowledge, confidence, or conclusions.
11. **Curiosity is persistent but evidence-sensitive.** Repeated unknown words, phrases, dialogue acts, concepts, and Why gaps accumulate intrinsic pressure across curricula. Time alone may raise urgency, but identical evidence must not trigger identical repeated searches.
12. **Mastery is self-assessed, never declared globally.** The learner measures character stability, grounded vocabulary and phrases, dialogue acts, tested predictions, causal gaps, and corroborated concepts. It pursues its weakest measured dimension and may claim mastery only within observed curricula and explicit evidence gates.
13. **A local model may be a conversation environment, not a teacher.** Noise constructs its own utterance from its mastery goal and curiosity ledger. Local-model replies create turn-taking and language observations with evidence score zero; they cannot satisfy lexical, conceptual, or causal gates without independent evidence.
14. **Global knowledge has one canonical owner.** Cross-curriculum curiosity and future shared beliefs live once in a global ledger. Per-seed states store only local observations and references to global priors; copying the entire global ledger into every experience is forbidden.
15. **Experiences are local; learned capability is global.** A seed report preserves what happened in one curriculum. Accepted lexical beliefs and accumulated event transitions live in one cross-curriculum memory, are loaded before the next experience, and are evaluated globally. Reports store only the delta from that prior.
16. **Capability requires a frozen external test.** Growing vocabulary, stored observations, rounds, and performance on a moving holdout are diagnostics, not capability growth. Benchmark sources and examples are frozen, excluded from training, and split by whole source into model-selection and untouched final sets.
17. **Choosing a representation is not evaluating it.** A parser, representation, or predictor selected on one set must independently beat its paired frequency baseline on the final set. Failure on either set retains the baseline and the failed candidate as a counterexample.
18. **Bounded-world success has zero real-world credit.** Procedural micro-world stages may demonstrate that a mechanism executes, but their scores cannot satisfy real-reading, language, abstraction-transfer, or causal gates.

## Forbidden substitutions

- Do not replace concept learning with embeddings from a pretrained model.
- Do not replace boundary induction with a hidden pretrained tokenizer while claiming autonomous discovery.
- Do not use an LLM judge as the ground truth for correctness.
- Do not call retrieval or fluent paraphrasing “understanding” without a predictive or grounding test.
- Do not improve apparent results by adding the answer page, target rule, exact vocabulary, or expected causal graph to the learner.

## Optional local-model boundary

```text
local model proposal (score 0)
        ↓
ordinary read-only search / environment test
        ↓
source and counterevidence evaluation
        ↓
AI_Noise belief update
```

If the helper is absent, slow, malformed, repetitive, or low quality, the pipeline continues with autonomous enumeration and search. A larger local model is not a remedy for a missing learning mechanism.

### Optional linguistic reference (morphological analyser)

`morphology_teacher.py` may wrap a Japanese morphological analyser (system `fugashi`/`MeCab`, system `janome`, or the wheel vendored under `experiments/_vendor/`). It follows the **same boundary as the local model**: its segmentation, part-of-speech tags, and dictionary forms are proposals at evidence score 0, not authority.

- It is used **only** in the developmental reading loop's *per-book* path (`japanese_reader_v1` → `record_reading`, the caregiver retelling), to correct verb dictionary forms and strip relative-clause fragments from subjects — the same slot as the LLM reading scaffold.
- It never touches the **frozen** benchmarks (`evaluate_comprehension`, `evaluate_retelling`), the character RNN, or `japanese_boundaries_v18`. Those only ever see the heuristic parse (`extract_story` defaults `use_teacher=False`).
- The autonomous word-boundary-discovery claim stays measured on the induction-only path. The analyser is a reference to check against, like a Wiktionary page — not a replacement for the discovery.
- `AI_NOISE_NO_MORPHOLOGY=1` forces the null backend; `test_architecture_contract` asserts the reading loop still extracts events without any analyser.

### Reading corpus (read-only, third-party)

The developmental reading loop fetches public text: **Aozora Bunko** (public-domain
literature), **Japanese Wikisource** (`イソップ童話集` + folktales), and **Tatoeba**
(`downloads.tatoeba.org` bulk export, CC-BY 2.0 FR — short example sentences bundled
into graded "readers"). See `experiments/ATTRIBUTION.md`. Tatoeba readers carry
`source: "tatoeba"`: they build vocabulary and character-model text but are **kept
out of the narrative benchmarks** (`evaluate_comprehension`, `evaluate_retelling`)
and graded by parse quality, not narrative comprehension. `japanese_word_meaning_v1`
(distributional, not narrative) *does* read the Tatoeba bundles — they are Noise's
own `heuristic_self` events and the corpus's richest source of concrete common nouns.

### Word meaning as a revisable belief (`japanese_word_meaning_v1`)

Every word's meaning is a belief with a confidence and a source trail (`beliefs[w]`),
folded onto a small closed coarse ontology (生き物 / 道具 / 場所 / …).

- **Testimony** — a ja.wiktionary / ja.wikipedia genus, and (when a local model is
  up) its answer to *one* closed-set discrimination question — enters the belief at
  a **capped weight** (`TESTIMONY_CAP`, currently 0.35). It is a hypothesis, held
  provisionally, and by invariant 10 it can never on its own make a word "understood"
  or satisfy any gate. Held-out test words are never researched and never asked.
- **Evidence** — Noise's own reading: how the word is *used* (acts under its own
  power → creature; only ever handled → thing; a destination → place) and the
  classes of the co-occurring words it *already understands*. Evidence is what
  moves confidence past the testimony cap and what a word must have before it
  counts as "understood".
- **Revision** — the belief is recomputed each cycle; when reading evidence
  outweighs an earlier testimony-only genus the belief flips and the change is
  kept in `revisions`. Counter-evidence can always demote "understood" (invariant 8).
- **Capability (frozen, held-out)** — does the belief machinery put a held-out
  word (genus-validated, concrete) in the right coarse class more often than naive
  co-occurrence propagation? One-sided significance over per-word gains; it is a
  measure of Noise's own inference, not of agreement with any source.

## Predict within the event, not the next event

The parsed corpus (isolated `subject|verb|object` clauses from real 19th-century
prose, coreference not resolved) contains no measurable signal for *next-event*
prediction: a verb bigram scores below the frequency baseline, and every context
key is seen with exactly one outcome. `event_structure_v1` therefore predicts
structure *inside* one event — `verb_cloze` (rank the observed verb among all
known verbs given its arguments) and `event_plausibility` (score a real event
above a corrupted one) — which do beat the baseline on a source-disjoint
holdout. `world_model_v51`, `association_learning_v33`, `causal_experiment_v28`
and `representation_learning_v31` (all next-event predictors) were retired.
Causal succession prediction is honestly unevaluated until the corpus carries
contrastive or interventional evidence (invariant 6).

## The learning stack below event_structure

Five modules feed or extend the predictor, each earning capability credit only
by the usual gate (beat the baseline on a frozen, source-disjoint split):

- **`coreference_v1`** — within-document pronoun/entity resolution before
  extraction, so an event sequence keeps one protagonist thread. Rule-based;
  no coreference-by-description.
- **`proposition_v1`** — turns discarded copular/possessive clauses into
  `entity|relation|value` propositions. Currently a data feed only (measured to
  carry no predictive signal on the present expository-heavy corpus).
- **`sequence_model_v1`** — a tiny from-scratch character RNN (hidden 24,
  ~4k params, no numpy, hand-written BPTT), time-boxed and incremental. Its
  held-out bits/char is a *continuous* capability signal and its `sample()`
  head is the generative substrate. It is not a substitute for a learning
  mechanism and carries zero credit until it beats the order-0 char baseline.
- **`active_curriculum_v1`** — retires closed-class curiosity gags ("in the")
  and turns the frozen model's held-out misses into search seeds, so discovery
  targets what the model gets wrong rather than what is frequent.
- **`capability_report_v1`** — one continuous multi-dimensional dashboard
  (per-task accuracy/lift/significance/coverage/trend, learning *efficiency* =
  lift slope vs training size, sequence-model perplexity, extraction health,
  boolean gates). Replaces "plateau: yes/no".

`llm_tooluse_v1` and `generative_dialogue_v1` are skill loops, not knowledge
paths: Noise formulates a checkable sub-task / composes an utterance, and the
local model's reply is scored only for *usefulness* (verified against Noise's
own grounded vocabulary and parser) or *comprehension* (was Noise understood).
The reply never updates a belief (invariants 10, 13).

## Decision replay, not state replay

`.local/events.jsonl` is an append-only, cross-module log (`{ts, curricula, module, event_type, before, after, reason}` per line) of discrete state-changing decisions: a benchmark locking, a selected model switching. It exists to answer "when and why did the system decide this" without trusting a human's memory of a status snapshot, and it coexists with (does not replace) each module's own bounded `revision_history`-style fields, which the algorithms themselves still read.

This is **decision replay, not full state replay**. Reading `events.jsonl` alone can reconstruct the *sequence of decisions* a module made and why. It cannot reconstruct the *evaluation numbers* behind those decisions (a `correct`/`total`/`lift` at some past point) — every module here recomputes those from scratch from the raw audit each call; nothing is derived solely from accumulated events. Reproducing a past number still requires re-running the owning module against the historical audit data, exactly as before this log existed. Do not describe this mechanism, or extend it, as if it captured a replayable world state — it captures a history of decisions about that state.

## Review gate

Before merging a version, answer:

- What did AI_Noise learn that was not directly encoded?
- How many experiences were required?
- What prediction could fail?
- What evidence would reverse the belief?
- Does the feature still work with the local LLM disabled?
- Are fluent output and genuine learned state clearly separated?
- Is the reported improvement measured on frozen examples that never selected the model?
- Does the paired comparison beat the exact baseline on both selection and final source sets?

These gates are exercised by `test_architecture_contract.py`, `test_event_structure_v1.py`, and `test_local_worker_v21.py`. Passing them prevents several known false claims, but it does not prove intelligence or semantic understanding.

If these questions cannot be answered, the version is not an advance toward the project objective.
