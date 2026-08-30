# Resource and usage policy

AI_Noise should spend computation and model usage only when it can change an evidence-backed decision.

## Test tiers

Use the quick profile after normal edits:

```bash
cd experiments
python3 run_tests.py --profile quick --quiet
```

It covers the current developmental-language and autonomous-web path without rerunning the older multi-seed evaluations. Use the full profile before a milestone push, after touching shared causal code, or for a release audit:

```bash
python3 run_tests.py --profile full --quiet
```

Do not repeatedly run the full profile when only documentation or one isolated parser changed.

## Web requests

- Read-only GET responses are cached under `.cache/web/` for seven days.
- The cache directory is ignored by Git.
- Set `AI_NOISE_DISABLE_CACHE=1` only when a fresh retrieval is required.
- Reports include cache hits, misses, and actual network-request counts.
- Candidate exploration stops when its explicit evidence target is met.
- Save full JSON to `--output`; use `--summary` during iterative checks.

## Optional local AI

Ollama may be used only as a proposal generator. The default helper model is `qwen3:4b`, overridable with `AI_NOISE_LOCAL_MODEL`. A local proposal has `verified=false` and `evidence_score=0.0`; it cannot update a belief until ordinary sources validate it.

Use local AI for bounded candidate generation, query variants, or compacting material that is already stored in the evidence ledger. Do not use it as the judge, source, or final answer. If its JSON is invalid, labels merely repeat the queried surface, or it is unavailable, return zero proposals and continue without it.

Larger local models should be loaded only after the 4B helper demonstrably fails on a task whose expected value justifies the extra VRAM and time.

## Codex task usage

Routine edits, targeted tests, and known fixes should use the default or lower reasoning effort. Increase effort for architecture changes, difficult diagnosis, security-sensitive work, or the final completion audit. Avoid subagents unless work is genuinely independent and parallelism saves more than the duplicated context costs.

Keep terminal output bounded. Prefer a short measured summary over printing complete reports into the conversation. The complete artifact remains on disk for inspection.

Official Codex documentation notes that higher reasoning effort can improve complex work but takes longer and uses more tokens; it recommends starting with the default effort and increasing it when needed: <https://learn.chatgpt.com/docs/models>.

ChatGPT Work and Codex share usage under the account plan, so unnecessary task runs consume the same overall allowance: <https://learn.chatgpt.com/docs/pricing>.

## Local-first operating split

Run routine learning with `python3 experiments/local_worker_v21.py run`. It invokes no Codex task and no remote model API. Inspect its compact heartbeat with `python3 experiments/local_worker_v21.py status`; do not ask Codex to poll ordinary progress. Request a safe stop with `python3 experiments/local_worker_v21.py stop`.

Use Codex only when the local worker reports an error, reaches a genuine architecture gap, needs a reviewed code change, or a milestone is ready to audit and push. Python evidence logic remains the decision-maker. Ollama 4B remains an optional zero-weight proposal generator, not a replacement judge.
