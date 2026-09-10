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
  `japanese_prediction_v1` adds one more channel: a word whose real story events
  Noise keeps rating implausible under its believed genus contributes a capped
  (≤0.30) penalty against that genus.
- **Genus classification** (2026-09-10) — a fine dictionary genus string is
  folded onto the **10** coarse classes (`生き物 / 人 / 身体 / 植物 / 食べ物 /
  道具 / 場所 / 自然物 / 出来事 / 気持ち`; the same tuple is mirrored in
  `cognition_v1.COARSE` and `japanese_dialogue_v1.COARSE`) by `_coarse`, in two
  passes: (1) the string ENDS with a class key — it is the definition's head
  noun ("鍵盤楽器" → 楽器 → 道具, "教育施設" → 施設 → 場所, "内臓の一つ" → 内臓 →
  身体), which is almost never a spurious substring; (2) a class key appears
  anywhere, minus a small `_COARSE_BLOCK` veto for the collisions the suffix
  pass does not catch ("末梢神経障害" contains 神). Rule order breaks ties. This
  is normalisation of an external source's wording, not an answer — a word is
  "understood" only on its own reading evidence. `_wiktionary_gist` has a
  last-run fallback and a wikipedia first-two-sentences fallback; a held-out
  probe whose genus fetch came back empty is **retried** (`REFS_MAX_ATTEMPTS`),
  not cached as a permanent exclusion (a transient fetch failure used to drop
  real concrete words — 太陽 / ランプ / お金 — from the pool forever, most of why
  it sat at ~19–20).
- **Capability (frozen, held-out)** — does the belief machinery put a held-out
  word (genus-validated, concrete) in the right coarse class more often than naive
  co-occurrence propagation? One-sided significance over per-word gains; it is a
  measure of Noise's own inference, not of agreement with any source.

### Knowledge → capability loop (`cognition_v1`, `capability_probe_v1`)

Reading accumulates knowledge; this loop turns it into problem-solving.
`AI_NOISE_COGNITION=0` disables it. Each reading cycle, over one concept from
the book just read:

1. **retrieve** — the concept's belief, the book's events mentioning it, and
   its understood neighbours (semantic, ranked by relevance).
2. **abstract** — `(subject_genus, verb, object_genus)` triples seen in ≥2
   books become rules with support / counterexamples; a failed application
   weakens a rule (invariant 8). Vacuous verbs (ある/いる/なる/する/…) never form a rule.
3. **generate** — problems it can auto-grade from its own verified data:
   `event_recall` (next verb in a book), `genus_recall`, `odd_one_out` /
   `common_property` (genus combination), `property_transfer` (predict an
   action from a cross-book rule — provably near-flat on this corpus, kept as
   an honest diagnostic).
4. **reason** — explicit retrieve / classify / compose steps, recorded; no LLM
   (invariants 10, 13). `わからない` when nothing supports an answer.
5. **self-evaluate** — was knowledge used? does the answer contradict a belief?
   is confidence calibrated? (This is not the correctness grade.)
6. **experience** — a Situation / Thought / Action / Result / Evaluation /
   Correction record, success and failure alike; the next attempt at a similar
   problem reads the Correction. Repeated failures are flagged.

**Cognitive Controller** (Phase 3): a contextual bandit picks the reasoning
*strategy* per problem type -- `lookup` (belief only), `compose` (neighbour
vote), `deliberate` (hypotheses + counter-evidence) -- from the loop's own
graded outcomes (Beta-smoothed verified-success rate per (type, strategy),
ε-greedy exploration, safest strategy wins a near-tie).  The frozen probe
re-solves with the controller's learned policy (no exploration), so the metric
tracks what the loop settled on.

**Capability is the frozen probe, not the rule count** (invariant 16).
`capability_probe_v1` freezes genus-combination problems whose **gold is the
independent ja.wiktionary genus** (from word_meaning's already-fetched gists —
no network), split into tiers *before* any gold is seen: each word is hashed to a
tier, and a problem lives in the tier of **the word it asks about** (`concept`),
so two problems about the same concept always share a tier — no concept /
dictionary entry straddles a split. A held-out-tier problem's *distractors* may
be challenge-tier (challenge makes no capability claim) or the same held-out
tier, never a different held-out tier.

- **challenge** — problems the frequency baseline fails, collected on purpose.
  Its before/after curve is a self-improvement DIAGNOSTIC; a rise here is *not*
  "beats a general baseline".
- **unbiased selection** — a salt sample of the whole population; gold and
  baseline correctness are never consulted at selection time (a baseline-correct
  problem is just as likely to be in it). Re-measured only when the model
  fingerprint moves (never re-rolled waiting for a lucky p-value). Diagnostic.
- **unopened final (+ reserve)** — never graded until the unbiased selection has
  cleared a **pre-registered** threshold (derive-rate + minimum effect) across
  two DISTINCT model fingerprints. Opened at most once each; an opened final is
  never re-graded; not used for learning, rules, corrections or the controller.

`capability_confirmed` requires ALL of: selection cleared the pre-registered bar
on two distinct models; an unopened final improved in the same direction; the
final beat its baseline by the minimum effect; tier separation is valid. The
probe reports `building` (with the reference-pool sizes and the explicit
bottleneck) until word_meaning's held-out concrete vocabulary is large enough to
fill the tiers — it never manufactures a signal from insufficient data. v2 kept
only baseline-fails problems and reported `lift = derive_rate` in disguise; the
reader archives the v2 state under `.local/audit/` and its rates are not
inherited.

### Judge a story event's plausibility, revise a concept when wrong (`japanese_prediction_v1`)

The Japanese reading loop accumulated a co-occurrence graph and dictionary
testimony but never made a falsifiable judgement about a story and revised when
it was wrong.  This module adds the predict → fail → self-correct step:

1. reading a real event `(subject, verb, object)` whose subject has a genus
   Noise believes, Noise scores its **plausibility** from its own concepts —
   abstracted `(subject_genus, verb_class, object_genus)` rules, the
   subject-genus / verb-class fit, the verb-class / object-genus fit.
2. it also scores a genus-**corrupted** version (verb class or object genus
   swapped).  Did Noise rate the real event above the fake, *more often* than a
   verb/argument-frequency baseline?  Tiered (`jb.Tiers`): SELECTION diagnostic,
   capability = a candidate checkpoint (anchor-streak 2) passing its one-shot
   unopened final.
3. a real event Noise keeps rating implausible is the self-correction signal: it
   accumulates per subject word, and a word whose story behaviour keeps looking
   wrong under its believed genus feeds `japanese_word_meaning_v1` a **capped
   (≤0.30) counter-evidence** against that genus — one channel in `_revise_belief`,
   never the sole cause of a revision (invariant 10).

**What this measured** (2026-09-10, live corpus): the same wall as English
`event_structure_v1`.  Predicting the *next verb class* from the subject's genus
is near-flat (secondary_diagnostic, ~+0.3pt over the marginal).  Plausibility
discrimination carries a little signal but is not yet significant (~+0.6pt full,
~+1.6pt on the ~22% of trials that carry any discriminative structure) —
because **90% of scorable events are `(creature subject, no object genus)`**, for
which every verb class is plausible.  The concept graph is ~78% 生き物 and the
concrete non-creature nouns that would make events checkable are exactly the
held-out-scarce set.  Narrative regularity, not a causal claim (invariant 6).
This is NOT the retired world-model predictors: genus-abstracted, baseline-gated,
frozen-tier, and its only learning output is capped genus counter-evidence.
`AI_NOISE_JA_PREDICTION=0` off.

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
- **`japanese_sequence_v1`** — the same RNN on Japanese, now instantiated as
  **two models by regime** (P1-1). (1) the **general** model (`jseq_clean_v4`,
  `.local/reading-sequence.json`) trains on the raw text of *every* book Noise
  read — Tatoeba bundles and books the parser could not break into events
  included — and is diagnostic-only + drives the display retellings. (2) a
  **dedicated narrative** model (`jnarr_seq_v1`,
  `.local/reading-narrative-sequence.json`) trains ONLY on recognised-narrative
  raw text (≥3 heuristic events, not Tatoeba, not vocab-fuel) minus every
  comprehension/retelling held-out collection. The retelling benchmark's
  likelihood model and `rnn_training_urls` are the **narrative** model, so its
  training source set is identical to the position baseline's by construction
  (the general model's corpus is a permanent superset of the narrative story
  set — measuring against it sat at `measurement_invalid_baseline_corpus_mismatch`
  forever).
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

### Say it from what you read (`japanese_dialogue_v1`)

`generative_dialogue_v1` composes from a plateaued English vocabulary and sits
at ~6%.  `japanese_dialogue_v1` composes a Japanese utterance from Noise's OWN
reading knowledge -- a word-meaning belief (`「椅子」は道具です。`), an abstracted
rule (`「きつね」ははしることがあります。`), a co-occurrence (`「鉄砲」は「おおかみ」と
いっしょに出てきます。`) -- and scores itself **behaviourally with echo removed**
(P1-2): a reply passes only if, after stripping everything it copied from Noise's
own sentence plus bare agreement markers, it still carries independent semantic
content AND that content is the *kind* of response the utterance's form asks for
(a question → a polarity/genus answer; a relation claim → info about the pair,
not a restatement; a genus/property claim → an independent use / attribute /
kind). A verbatim or partial echo, a leaked `学習者:` prompt prefix, a bare
「そうですね」, a reply that names the concept but says nothing about it, an
asserted contradicting genus, and a helpful guess at a sentence Noise could not
ground all **fail** — each recorded separately (`echo_detected`,
`clarification`, `relevant_new_information`, `response_to_requested_act`,
`contradicts_belief`, `partner_guessed_malformed`) alongside the utterance's own
quality (`parseable`, `belief_supported`, `relation_supported`, `malformed`).
It is a **use-test**: a belief Noise cannot put into a sentence a person
understands is not yet usable. Capability: a frozen set of **tasks** — concept
*and* a fixed utterance intent, so a shift in the learned best strategy cannot
move the frozen metric — one utterance per cycle alternating a rotating practice
concept and a frozen probe task (a full round = len(frozen) cycles; the round's
understood-rate + echo/clarification/malformed rates are tracked). v2 counted
echoes as understood; the reader archives the v2 state under `.local/audit/` and
its rates are not inherited. `AI_NOISE_JA_DIALOGUE=0` off.

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
