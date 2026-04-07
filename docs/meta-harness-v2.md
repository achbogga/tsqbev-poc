# Meta-Harness V2

`harness_v2` is the parallel research control plane for this repo. It does not replace the
incumbent supervisor immediately. Instead, it searches over **single-file Python harnesses** and
benchmarks them on replay tasks before they are allowed to influence live GPU work.

## Candidate Contract

Every candidate is a single file named `candidate.py` and must define:

- `CANDIDATE_METADATA`
- `run_harness(task: dict) -> dict`

The result dict must include:

- `objective`
- `targeted_bottleneck`
- `priority_tags`
- `suppress_tags`
- `force_priority_only`
- `kill_conditions`
- `rationale`
- `retrieval_queries`
- `worker_routes`
- `report_outline`
- `context_summary`

## Scorecard

Candidates are scored out of `100`:

- `25` decision quality
- `20` failure diagnosis
- `15` retrieval quality
- `15` execution correctness
- `10` efficiency
- `10` publication quality
- `5` live shadow behavior

Promotion is gated:

- benchmark score must beat the incumbent by at least `8`
- execution correctness must not regress
- shadow result must stay promotable

## Commands

- `tsqbev harness-benchmark`
- `tsqbev harness-search`
- `tsqbev harness-shadow --candidate-path ...`
- `tsqbev harness-promote --candidate-path ...`
- `tsqbev harness-report`
- `tsqbev harness-memory-sync`

Unless `--artifact-dir` is provided, these commands resolve to `artifacts/harness_v2`.

## Remote/Local Split

- remote reasoning roles: proposer, critic, judge
- default provider order: `MCP` if configured, then `OpenAI`, then heuristic fallback
- local side-effectful roles: benchmark replay, memory sync, research-loop execution, publishing

`MCP` access is configured through `TSQBEV_HARNESS_MCP_CMD`, which should accept JSON on stdin and
return text on stdout.

## Memory Integration

`harness_v2` emits summary artifacts with kinds:

- `harness_benchmark`
- `harness_shadow`
- `harness_promotion`
- `harness_context_summary`

These are ingested by the existing research memory stack. Context summaries are written whenever
assembled candidate context exceeds `50%` of the configured prompt budget.
