# tsqbev-poc Program

Status: enabled.

Reference workflow template:

- Andrej Karpathy `autoresearch`: <https://github.com/karpathy/autoresearch>

This repo is allowed to run a bounded local research loop.

Current priorities:

1. tune a public `nuScenes v1.0-mini` object baseline
2. fix routing and seeding pathologies before spending more epochs
3. bootstrap the student with an external pretrained LiDAR teacher through caches
4. record keep/discard decisions for a small fixed recipe set
5. update docs with measured mini-baseline and teacher-bootstrap results

Loop contract:

- dataset scope: `nuScenes v1.0-mini` only
- split scope: `mini_train` / `mini_val`
- max recipes per invocation: `3`
- recipe changes allowed:
  - sparse query-budget allocation across LiDAR / proposal / global seeds
  - image-backbone family among compact pretrained backbones
  - frozen vs unfrozen pretrained image backbone
  - learning rate
  - batch size
  - gradient accumulation
- real acceptance criteria:
  - higher official `mini_val` `NDS`, then `mAP`
  - source mix remains genuinely multimodal instead of collapsing to one source
  - lower validation loss only as a tiebreaker
  - no runtime errors
  - measured forward latency captured in the ledger
- output artifacts:
  - `research_loop/results.jsonl`
  - `research_loop/summary.json`
  - per-recipe `mini_val` export and evaluation output
  - per-recipe source-mix diagnostics

Forbidden:

- unbounded mutation loops
- full `v1.0-trainval` search via the research loop
- indefinite training runs
- silent rewriting of core contracts
- deleting experiment evidence after a run completes
