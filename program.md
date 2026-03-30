# tsqbev-poc Program

Status: enabled.

Reference workflow:

- Andrej Karpathy `autoresearch`: <https://github.com/karpathy/autoresearch>
- baseline `program.md`: <https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md>

This repo is allowed to run a bounded local research loop, but only under the stricter contract
below.

## Setup

Before any invocation:

1. verify that the active dataset is `nuScenes v1.0-mini`
2. verify that `program.md` still says `Status: enabled.`
3. treat the repo as evidence-first and append-only
4. preserve all prior experiment artifacts
5. if W&B is available, mirror the run there under the stable entity `achbogga-track`

## In-Scope Surface

The bounded loop is intentionally narrower than the official `autoresearch` baseline.

Allowed mutation surfaces:

- sparse query-budget allocation across LiDAR / proposal / global seeds
- compact pretrained image-backbone family
- pretrained image-backbone freeze policy
- learning rate
- batch size
- gradient accumulation
- optional cached external LiDAR teacher seed replacement

Read-only surfaces during the loop:

- dataset contracts
- evaluation harness semantics
- public metric definitions
- deployment/export contracts
- scale-gate thresholds

## Research Org Contract

Follow the strongest transferable ideas from `karpathy/autoresearch`, adapted to this repo:

- establish or re-check one incumbent first
- keep a fixed, comparable mini-dataset budget
- log every run to an append-only ledger
- record the hypothesis and mutation reason for every run
- advance only when official metrics improve
- preserve failed runs in the ledger instead of deleting evidence

The active local loop shape is:

1. baseline or carry-over incumbent recheck
2. bounded exploration on `mini_train / mini_val`
3. bounded exploitation around the best measured incumbent
4. promote exactly one record per invocation

## Dataset And Budget

- dataset scope: `nuScenes v1.0-mini` only
- split scope: `mini_train` / `mini_val`
- max recipes per invocation: `5`
- fixed comparable train budget per recipe: `max_train_steps = 960`
- recipe budget shape:
  - `1-3` baseline/exploration recipes
  - up to `2` exploitation recipes derived from the current incumbent

## Acceptance Logic

Primary selection metric:

- higher official `mini_val` `NDS`

Secondary metrics:

- higher official `mini_val` `mAP`
- lower validation loss only as a tiebreaker

Required discipline:

- routed query bank must remain genuinely multimodal
- record synthetic forward latency for every completed recipe
- teacher-assisted recipes must be treated as paired ablations, not prose
- teacher-assisted recipes count only if teacher-cache coverage has been audited first
- W&B logging is advisory and must never affect the accept/reject decision path
- do not call a recipe scale-ready just because it wins the local sweep

## Required Artifacts

Every invocation must write:

- `research_loop/results.jsonl`
- `research_loop/results.tsv`
- `research_loop/summary.json`
- per-run `manifest.json`
- per-run official `mini_val` export and evaluation output
- per-run source-mix diagnostics

If W&B is available, each invocation must also mirror the same metrics and metadata there under the
project derived for that architecture family. Hyperparameter and performance tuning stays grouped in
the same W&B project; materially different architecture families use different W&B projects.

The ledger must distinguish:

- interim `advance/reject/crash`
- final `promote/discard/crash`

There must be exactly one final promoted record per completed invocation.

## Forbidden

- unbounded mutation loops
- full `v1.0-trainval` search through the research loop
- silent rewriting of core contracts
- deleting experiment evidence after a run completes
- treating surrogate loss alone as promotion evidence
- authorizing 10x compute without clearing `specs/005-scale-gate-contract.md`
