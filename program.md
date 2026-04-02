# tsqbev-poc Program

Status: enabled.

Reference workflow:

- Andrej Karpathy `autoresearch`: <https://github.com/karpathy/autoresearch>
- baseline `program.md`: <https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md>

This repo is allowed to run a bounded local research loop, but only under the stricter contract
below.

## Standing Research Directives

These instructions are durable repo policy, not one-off chat guidance.

- challenge every material design decision from first principles before extending it
- treat the current design as provisional until local evidence supports it
- move fast with bounded changes; do not spend large effort on unfalsifiable ideas
- browse primary papers, official repos, and official weights for unstable SOTA areas such as KD,
  pretrained backbones, query design, multimodal fusion, and deployment tradeoffs
- treat KD as a broad menu, not a single method: logits, feature, `1x1` alignment, relational,
  dense output targets such as heatmaps / BEV maps / segmentation maps, online, mutual,
  self-distillation, and teacher-anchor transfer are all in scope
- for the current repo, prioritize KD targets by ROI:
  1. ranking-critical teacher outputs
  2. lightweight feature alignment
  3. dense maps such as heatmaps or BEV segmentation targets
  4. relational KD
  5. online, mutual, or self-distillation
- if a public dense-BEV upstream stack is better supported than the current custom path, pivot to
  it rather than deepening the custom sparse-query line
- treat BEVFusion, OpenPCDet, BEVDet / BEVDepth, MapTRv2, EfficientViT, DINOv2 / DINOv3, and
  MIT HAN Lab compression methods as first-class candidates
- prefer the highest-ROI falsifiable change first, not the most fashionable or largest one
- fix small blockers immediately when they are clearly slowing the loop
- do not stop after one run if the next step is clear and bounded
- stop and reassess when the direction becomes a rabbit hole
- hydrate the local research-memory brief before planning the next bounded move
- publish PI-readable reports and machine-readable sync artifacts after each invocation

## Token-Burn Discipline

Every nontrivial direction must maintain a lightweight `token_burn_score`:

- `expected_roi`: `1-5`
- `integration_cost`: `1-5`
- `uncertainty`: `1-5`
- `evidence_gain`: `1-5`
- `token_burn_score = integration_cost + uncertainty - expected_roi - evidence_gain`

Interpretation:

- `<= -2`: proceed aggressively
- `-1` to `2`: proceed, but checkpoint after the next bounded result
- `>= 3`: stop and reassess before spending more time or compute

Rabbit-hole boundary:

- two bounded failures on the same hypothesis without new evidence
- measured bottleneck changed and the branch no longer targets it
- implementation complexity is rising faster than evidence quality
- `token_burn_score >= 3`

## Setup

Before any invocation:

1. verify that the active dataset is `nuScenes v1.0-mini`
2. verify that `program.md` still says `Status: enabled.`
3. treat the repo as evidence-first and append-only
4. preserve all prior experiment artifacts
5. if W&B is available, mirror the run there under the stable entity `achbogga-track`
6. refresh the local research-memory state and pre-run brief before the next invocation

## In-Scope Surface

The bounded loop is intentionally narrower than the official `autoresearch` baseline, and the
active architecture search target is now a dense-BEV fusion stack rather than the legacy
sparse-query prototype.

Allowed mutation surfaces:

- LiDAR BEV encoder family
- camera BEV encoder family
- BEV fusion trunk choice
- detection head choice
- lane / map head choice
- compact pretrained image-backbone family
- pretrained image-backbone freeze policy
- learning rate
- optimizer schedule
- gradient clip norm
- batch size
- gradient accumulation
- best-checkpoint selection policy
- detection loss mode and hard-negative budget
- bounded score-threshold / top-k calibration
- optional cached external LiDAR teacher guidance, including full seed replacement
- teacher-anchor selection policy inside the fixed seed budget

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
- record the active bottleneck and token-burn score for each direction under investigation
- keep the legacy sparse-query line only as a bounded comparison control unless the dense-BEV
  reset is clearly worse on the same evidence

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
- exported predictions must remain geometrically sane in count and range
- overfit-mode runs must evaluate the selected checkpoint, not only the last checkpoint
- ranking-sensitive runs must record the bounded score-threshold / top-k calibration sweep
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
- per-run calibration summary when multiple thresholds or `top_k` values are tried
- per-run source-mix diagnostics
- per-run root-cause verdict
- `artifacts/memory/sync_manifest.json`
- `artifacts/memory/brief.json`
- `docs/reports/current.md`

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
