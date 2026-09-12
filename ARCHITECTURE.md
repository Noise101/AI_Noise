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
10. **Optional local AI has no vote.** A local model may propose bounded candidates or queries. Its output always begins unverified with evidence score zero and cannot directly update knowledge, confidence, or conclusions. (A deterministic morphological analyser's *segmentation/POS/lemma* is parse infrastructure, not a proposal about meaning, and is exempt on that basis — see "Structural morphological analysis vs. semantic proposal" below; any *interpretive* suggestion it makes still falls under this invariant.)
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

### Structural morphological analysis vs. semantic proposal (morphological analyser)

`morphology_teacher.py` may wrap a Japanese morphological analyser (system `fugashi`/`MeCab`, system `janome`, or the wheel vendored under `experiments/_vendor/`). Two of its uses sit on **opposite sides** of invariant 10, and the difference is what is being trusted, not which library provides it:

- **Structural parse infrastructure (permitted to feed learning).** A deterministic tokenizer's segmentation, part-of-speech tag, inflection form, and dictionary (lemma) form carry no claim about the story's *meaning* — they are the same category as an English whitespace tokenizer or a regex sentence splitter. `japanese_event_v1.extract_story` uses the analyser (when installed and `AI_NOISE_NO_MORPHOLOGY` is not set) to recover the case-marked subject/verb/object structure of a sentence — particularly multi-clause and relative-clause sentences the pure particle heuristic cannot break down — and falls back to the particle-only heuristic per sentence when the analysis is degenerate (see `_morph_sentence_events`'s quality gates: a kana-only sentence the dictionary cannot lemmatise, an out-of-dictionary proper noun glued into a nonsense verb, an implausible subject fragment, or a chain producing mostly-unnamed subjects). Because this is structure recovery, not a semantic claim, its output is stamped `heuristic_self` like the pure-heuristic parse and **is** Noise's own experience — it may enter `events_store`, the frozen benchmarks, the character RNN, and word-meaning evidence, exactly as the particle heuristic always could. Every judgement about what the recovered structure *means* — a word's genus, an event's plausibility, coreference by description, retelling coherence — still runs entirely on Noise's own heuristics and reading evidence; the analyser never supplies or corroborates a belief about meaning.
- **Semantic reference (stays score 0, never authority).** Anywhere the analyser's or a local model's *interpretation* is used directly — the older per-book teacher refinement (`refine_with_teacher`, `use_teacher=True`) that suggests a corrected verb form as a proposal rather than deriving it from case roles, and the LLM reading scaffold — it follows the same boundary as the local model in the diagram above: evidence score 0, corroboration required, never a direct belief update.
- The autonomous word-boundary-discovery claim stays measured on the induction-only path (`japanese_boundaries_v18`); the analyser is a reference to check against there, like a Wiktionary page, never a replacement for the discovery.
- `PARSER_VERSION` bump on a structural-parse change clears `events_store` and re-walks `shelved_stuck`/`unparsable` books back into rotation (`reevaluate_stale_parses`) — an extraction upgrade does not inherit an old benchmark pass, it re-earns one under the new parse.
- `AI_NOISE_NO_MORPHOLOGY=1` forces the null backend; `test_architecture_contract` and `test_japanese_event_v1.MorphologyDrivenParseTest` assert the reading loop extracts events (via the particle heuristic alone) with no analyser installed, and that the analyser-driven path recovers structure the heuristic alone cannot.

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

`japanese_proposition_v1` complements action events with conservative ordinary
stative clauses: `猫は動物です`, `レモンは黄色い`, `犬には足がある`, and
`本は机の上にある`.  It uses only structural morphology (or a smaller regex
fallback), rejects multi-clause/ambiguous predicates, and records each assertion
as a revisable `proposition_self` observation.  An assertion is not truth: direct
class evidence requires repetition in distinct read sources, and contradictory
or negative observations remain available for revision.  The per-cycle context
counters are rebuilt from the distinct current corpus rather than incrementing
again when the same corpus is replayed; rereading one book cannot manufacture
independent support.

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
- **`semantic_representation_v1`** — a dependency-free 20-dimensional
  skip-gram model trained incrementally on subject/object roles, verbs and
  co-occurrences from `heuristic_self` events only.  It is the first shared
  learned representation between reading and cognition: nearest grounded
  neighbours may propose a coarse class to the cognitive controller, while
  dictionary/LLM genera are never prediction targets.  Re-read source
  fingerprints prevent duplicate experience, parser-version changes reset the
  derived vectors, and the anchor holdout is diagnostic only.  Capability is
  still awarded solely by `capability_probe_v1`'s source/tier-separated gate.
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

**The production/comprehension gap, and speaking about a "roughly understood"
word (v5).** Reading tolerates a partially-resolved belief while still tracking
narrative structure; `_claim()` originally required a *confirmed* genus
(`understood=True`, `confidence>=0.55`) before producing ANY utterance at all —
harmless for reading, but a hard wall for production, since only ~16% of
encountered vocabulary ever clears that bar. A fifth, genus-independent
strategy, `"usage"`, closes part of this gap the way a person actually
speaks: it reports an *actually-read* verb usage straight from
`wm["profiles"][word]` (`「いたち」がはしるのを読んだことがあります。`) rather than a
claim about what the word means, so it needs no confirmed genus — a "roughly
understood" word is still something Noise has seen used, and using it is
itself the observation, not an assertion of understanding (`belief_supported`
is always true for a real usage). To keep the **frozen probe** comparable
across the version bump, `"usage"` is deliberately excluded from
`_frozen_strategy`'s strategy set and from `_understood_concepts` (the frozen
pool's only source) — it widens only the **practice** rotation, via a
separate `_usage_concepts()` pool of profiled-but-genus-unconfirmed words.

**Self-correction from conversational failure (`dialogue_feedback`).** The
owner's framing: humans use words they have only roughly understood, and
correct them once they find out they were wrong. `_record_outcome` tracks
hits/misses per word+assumed-genus across dialogue turns (usage turns are
skipped — they assert no genus, so a misunderstanding there is not evidence
against a genus belief); `build_feedback` turns a persistent miss run into
capped counter-evidence, in the exact shape `japanese_prediction_v1
.build_feedback` already produces. `japanese_reader_v1` merges both into ONE
`prediction_feedback` argument to `word_meaning.learn_and_evaluate` (higher
`strength` wins per word on overlap) — one revisable-belief channel
(invariant 10), now fed by two first-person sources: prediction-from-reading
and failure-from-speaking.

### Direct, meaningful conversation (`noise_chat_v1`)

`local_worker_v21.py talk "..."` and the local GUI use a stateful, bounded
dialogue engine rather than a stateless greeting/regex loop. Noise records an
explicit interpretation for every turn, carries the current topic and pending
question across turns, composes greetings from actual conversation/reading
state, separates shortened-question modifiers from their head noun, and can
report what it remembers, what it merely heard, and what it has independently
grounded. A user can inspect the interpretation in the GUI and explicit parsing
feedback is retained as an error so the same reading is not silently presented
as established. Noise can also ask about an unknown topic, remember an owner's
explicit claim or preference, recall it later, and retain an explicit
correction as an error. A claim is stored as `owner_testimony` with
`conversation_memory_not_world_fact`; it never enters the word-meaning ledger,
the semantic vectors, rules, or capability probes merely because a human said
it.  Contradicted conversation claims are retracted but retained in history.
Replies are composed only from these typed states and bounded clauses. Until
generation itself becomes reliably meaningful, an honest grounded response or
question is preferred to fluent-looking nonsense; dialogue fluency is not
reported as learned world knowledge.

Whether an utterance FUNCTIONS as a question is decided structurally
(`_is_question_form`), not by the presence of a written `？` -- ordinary
Japanese speech and casual text routinely end a real question in
です か/ますか/でしょうか, a bare 終助詞 か or の, or the colloquial かな/かしら,
and omit the mark entirely (の is ambiguous with the attributive/nominalising
連体化 の -- only the analyser tells them apart, so a bare の is read as a
question only when it is confirmed 終助詞, and never guessed at all with no
analyser installed). Relying on `？` alone silently dropped very common
questions ("元気ですか", "あなたの名前は") into the unresolved fallback. A
sentence-final remark/reaction particle (ね/よね/なあ -- "今日はいい天気です
ね") is a phatic remark, not testimony, and is never stored as a claim even
when it grammatically parses as one, and even when a prior turn is waiting for
an answer (a remark following a pending question is not assumed to be that
answer). A stored claim's predicate is rejected if it is itself question-shaped
-- a loose regex match must not turn the user's own unanswered question back
into "testimony" about its subject. The unresolved fallback asks the user to
rephrase rather than declaring the utterance unreadable.

**Small-talk intents are keyed by morphological lemma + structure, not a
literal surface string.** Matching one exact spelling, conjugation, or
politeness level ("ありがとうございます") is a permanent source of live
failures on ordinary variation -- casual spelling ("こんばんわ" for こんばんは),
politeness level ("ありがとうございました" / "ありがとうね" / "どうも
ありがとうございます"), or a different conjugation of the same predicate
("できます" / "出来るの", both lemma できる). `_is_greeting`, `_is_thanks`,
`_is_wellbeing_question`, `_is_capability_question`, and
`_is_identity_question` check the analyser's dictionary-form lemma and part
of speech (an interjection's lemma is in a small closed set, a predicate's
lemma is できる regardless of how it was conjugated, a leading pronoun plus a
名前/誰 lemma) rather than anchoring a regex to one written form of the
sentence. Each keeps its original literal-regex behaviour as the fallback
when no analyser is installed (`AI_NOISE_NO_MORPHOLOGY=1`) -- narrower
coverage without the analyser is expected and acceptable; a wrong or
fabricated classification is not. Every remaining intent in `_interpret`
(correction, parse feedback, recall of knowledge/preference, curiosity,
learning/activity status, Noise's-own-preference, confirm-understanding,
preference, "Xとは何", and the generic claim topic/predicate split) follows
the same lemma-or-structure-first design, each with its literal regex kept
only as the no-analyser fallback -- there is no remaining classifier in this
module anchored to one exact written form as its primary path.

A regex anchored to a written form can also fail the OTHER way: matching a
substring that is not the token it looks like. `_topic`'s original explicit-
marker regex matched the two characters "って" wherever they occurred,
including embedded inside an unrelated verb's て-form ("何を知っている" ->
"知って" contains って in the middle of one word, not the quotative って
particle) -- silently producing a garbage topic ("何を知"). The fix finds は/
って/とは as their OWN 助詞 token via the analyser, not a character search, so
a token boundary is required, not merely a matching substring.

**Multi-topic memory (`topic_stack`).** A single `current_topic` string cannot
represent "go back to what we were discussing before" -- every topic switch
overwrote it. `_touch_topic` maintains a capped, most-recent-last recency
stack; `follow_up` ("もっと教えて", "それについては？") expands on the topic
still on top without repeating an already-surfaced fact, and `go_back_topic`
("さっきの話に戻って", "犬の話に戻って") resumes an earlier one from the stack,
generic or by name, and says plainly when the named topic was never discussed
rather than starting one.

**Recall answers the asked subject, not every subject Noise holds a claim
about.** "猫について何を知っている？" must answer about 猫, not concatenate
every unrelated claim Noise happens to hold (a real failure: a question about
one topic produced a run-on sentence mixing claims about three others). When
the question names a subject, `_recall_knowledge` answers about that subject
alone (or says plainly it has heard nothing about it); only a bare "何を知っ
ている？" with no named subject lists recent memories in general.

**A bare pronoun (それ/これ/あれ/私/あなた) is never stored as a claim's
subject key.** These have no stable referent in a stored claim -- whose "私"
it is (the user's, or Noise's) is not resolvable from the string alone, and
"それは覚えることではない" (the user objecting to what Noise just stored) must
not itself become a claim keyed literally "それ". `japanese_proposition_v1`
rightly allows 私 as a subject for general narrative text, where it is a
stable first-person narrator; in direct conversation it is deixis, and
`_claim_from_text` rejects it before either extraction path can store it.
The same deixis applies to greeting register: `_greeting` echoes the
time-of-day the USER greeted with, not the server's own wall clock -- replying
"こんにちは" to their "こんばんは" because the two disagree reads as not having
heard them.

**Local model as a presentation layer, not a content source
(`PhrasingModel` / `_phrase_naturally`).** Once a reply's content is fully
decided by the ordinary rule logic above, an available local model MAY
re-express it in more natural Japanese -- but this sits on the same side of
the local-model boundary as the induction-only analyser use above, not the
belief-affecting side: every content word (kanji/katakana run) of the
template reply must survive verbatim in the rephrase, and the length must
stay in a sane range, or the original template is used unchanged. The model
can change how something is said; it can never add, drop, or alter what is
said. The template is retained (`turn["template_reply"]`) whenever a rephrase
replaces it, so what Noise actually decided stays auditable.
`AI_NOISE_CHAT_PHRASING=0` disables this layer entirely.

**Self-trained associative recall memory (`conversation_embedding_v1`).** A
small, dependency-free skip-gram embedding -- the same technique and the same
score-0 role as `semantic_representation_v1` on the reading side, applied to
`noise_chat_v1`'s own claims and `topic_stack` instead of read events (not
replacing concept learning with a pretrained embedding: it is trained only on
Noise's own conversation history, from nothing). When an unfamiliar topic
comes up again, the nearest already-discussed topic (restricted to topics
Noise can truthfully say it has talked about) is offered as a recall
*question* -- "we talked about X before, is this related?" -- never an
assertion that the two are the same concept, and it never writes, merges, or
changes a claim by itself; the human's answer is what would, through the
ordinary claim path, become testimony. A topic mentioned for the very first
time has no embedding yet and legitimately produces no recall -- there is
nothing to be similar to; the memory only helps from the second time a
related topic surfaces onward, and the same recall is never repeated for the
same subject once offered.

**Read-only web reference lookup (`_web_gist`).** When Noise does not know a
topic and is about to ask about it (`_ask_about`, first ask only, and only
when no live claim already covers the subject), it may also look the word up
via the SAME cached, network-budgeted Wiktionary/Wikipedia fetch the reading
loop uses (`japanese_word_meaning_v1._wiktionary_gist` /
`_wikipedia_genus`) -- one client, one cache, one budget policy, not a second
implementation. A found gist is stored as a claim with `source:
"web_reference"` (distinct from a human's `owner_testimony`, same
`evidence_role: conversation_memory_not_world_fact`): capped, revisable
testimony, never written into the word-meaning belief store this module does
not touch, and never presented as Noise's own confirmed understanding --
the reply says plainly "まだ自分では確かめていません". `AI_NOISE_SKIP_WEB_LOOKUP=1`
disables it; `web_lookup` is injectable for tests.

**A fetch that happened is not discarded just because it did not resolve
cleanly.** Per the owner ("人間はみたものを捨てない" -- a person does not throw
away what they have seen): the first version of the lookup above discarded
the ENTIRE result whenever `_wiktionary_gist` could not extract a clean
genus, even when a real page had been fetched and read (a definition
sentence, related terms). `_wiktionary_gist` now also returns the raw
fetched sentence (`gist_text`); `_web_gist` falls through genus -> raw
definition -> related terms before giving up, and anything found this way is
kept as an **encounter** (`_remember_encounter`, `state["encountered"]`) --
distinct from a `claim` (a specific subject/predicate assertion someone can
confirm or retract): an encounter carries no assertion shape at all, just
"I have come across this, unconfirmed". Neither a claim nor an encounter is
ever promoted to Noise's own confirmed understanding by being stored.

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
