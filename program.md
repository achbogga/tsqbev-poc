# tsqbev-poc Program

Status: enabled.

Reference workflow template:

- Andrej Karpathy `autoresearch`: <https://github.com/karpathy/autoresearch>

This repo is allowed to run a bounded local research loop.

Current priorities:

1. tune a public `nuScenes v1.0-mini` object baseline
2. record keep/discard decisions for a small fixed recipe set
3. run official local `mini_val` export/eval for the selected recipe
4. update docs with measured mini-baseline results

Loop contract:

- dataset scope: `nuScenes v1.0-mini` only
- split scope: `mini_train` / `mini_val`
- max recipes per invocation: `3`
- recipe changes allowed:
  - batch size
  - gradient accumulation
  - learning rate
  - frozen vs unfrozen pretrained image backbone
- real acceptance criteria:
  - lower validation loss on `mini_val`
  - no runtime errors
  - measured forward latency captured in the ledger
- output artifacts:
  - `research_loop/results.jsonl`
  - `research_loop/summary.json`
  - best-checkpoint export and `mini_val` evaluation output

Forbidden:

- unbounded mutation loops
- full `v1.0-trainval` search via the research loop
- indefinite training runs
- silent rewriting of core contracts
- deleting experiment evidence after a run completes
