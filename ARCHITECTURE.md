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

## Review gate

Before merging a version, answer:

- What did AI_Noise learn that was not directly encoded?
- How many experiences were required?
- What prediction could fail?
- What evidence would reverse the belief?
- Does the feature still work with the local LLM disabled?
- Are fluent output and genuine learned state clearly separated?

If these questions cannot be answered, the version is not an advance toward the project objective.
