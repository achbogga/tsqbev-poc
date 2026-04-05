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
- write durable literature syntheses and technique databases under `research/knowledge/` so they
  become queryable memory rather than one-off notes
- treat KD as a broad menu, not a single method: logits, feature, `1x1` alignment, relational,
  dense output targets such as heatmaps / BEV maps / segmentation maps, online, mutual,
  self-distillation, and teacher-anchor transfer are all in scope
- for the current repo, prioritize KD targets by ROI:
  1. ranking-critical teacher outputs
  2. lightweight feature alignment
  3. dense maps such as heatmaps or BEV segmentation targets
  4. relational KD
  5. online, mutual, or self-distillation
- keep the lane track methodical:
  - establish a real OpenLane baseline and evaluation artifact first
  - do not mix lane into the active detection loop until detection is no longer bottlenecked by
    ranking and source-mix collapse
- if a public upstream stack or teacher suite is better supported than the current custom path,
  pivot to it rather than deepening the custom sparse-query line
- treat Sparse4D, BEVFormer v2, OpenPCDet, BEVFusion, MapTRv2, EfficientViT, DINOv3, SAM 2.1,
  Alpamayo, and MIT HAN Lab compression methods as first-class candidates
- treat Alpamayo as a teacher/evaluator and long-tail mining tool, not as the in-vehicle runtime
  perception trunk
- prefer hosted frontier planner and critic models for the research control plane whenever
  credentials exist, with local fallback only on failure
- prefer the highest-ROI falsifiable change first, not the most fashionable or largest one
- fix small blockers immediately when they are clearly slowing the loop
- do not stop after one run if the next step is clear and bounded
- stop and reassess when the direction becomes a rabbit hole
- hydrate the local research-memory brief before planning the next bounded move
- publish PI-readable reports and machine-readable sync artifacts after each invocation
- treat lane work as an explicit secondary bootstrap track: prove the isolated OpenLane lane path,
  export, and evaluation first, then consider mixing it into the main detection loop

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
active architecture search target is now a foundation-teacher perspective-sparse student rather
than either the legacy sparse-query prototype or a pure dense-BEV runtime.

Allowed mutation surfaces:

- runtime camera foundation backbone and projection policy
- perspective auxiliary head choice and weighting
- sparse temporal aggregation family
- LiDAR anchor prior family
- dense-BEV teacher or control-arm choice
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
- activation checkpointing and GPU auto-fit policy for large frozen teachers

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
- emit a `boss_progress_verdict` that compares the promoted run against the previous incumbent and
  the best historical mini result
- if repeated `incremental_progress`, `schedule_checkpoint_drift`, or regression appears, suppress
  low-ROI exploit families automatically instead of reopening them by habit
- preserve failed runs in the ledger instead of deleting evidence
- record the active bottleneck and token-burn score for each direction under investigation
- keep the legacy sparse-query line only as a bounded comparison control unless the current reset
  is clearly worse on the same evidence
- if two consecutive winner-line continuations fail to produce a meaningful improvement, stop
  nudging weights and pivot to a new architecture or teacher family backed by fresh literature
- the default pivot order is:
  1. DINOv2/DINOv3 feature projection into the camera branch
  2. BEVFormer v2-style perspective supervision
  3. stronger BEV-space distillation from OpenPCDet / BEVFusion teachers
  4. staged lane/map integration through MapTRv2

The active local loop shape is:

1. baseline or carry-over incumbent recheck
2. bounded exploration on `mini_train / mini_val`
3. bounded exploitation around the best measured incumbent
4. promote exactly one record per invocation

## Continuous Supervisor

The bounded loop is not enough by itself. A Karpathy-style lab workflow in this repo now means:

1. one bounded invocation runs at a time on the active GPU
2. a supervisor waits for any external run to finish instead of contending blindly
3. before each invocation, the supervisor must write a first-principles checkpoint that records:
   - current strongest evidence
   - active bottleneck
   - smallest bounded next move
   - stopping condition for the branch
4. after each invocation, the repo must sync memory, rebuild the PI brief, append the ledger,
   and publish the generated report artifacts
5. the next invocation should start automatically unless a stop file is present or the repo is
   explicitly disabled
6. the next invocation must read the prior `boss_progress_verdict` and obey its
   `boss_policy_next`; suppressed exploit families stay suppressed until the evidence changes

The intended entrypoint is `tsqbev research-supervisor`, not manual one-off screens, whenever the
goal is continuous unattended progress.

Separate from the GPU research supervisor, the repo should keep a CPU-only daily maintenance agent
alive via `tsqbev maintenance-supervisor`. That process is responsible for repo hygiene checks,
memory sync, PI-facing maintenance reports, and tech-debt triage without contending for the active
training GPU.

## Dataset And Budget

- dataset scope: `nuScenes v1.0-mini` only
- split scope: `mini_train` / `mini_val`
- max recipes per invocation: `7`
- fixed comparable train budget per recipe: `max_train_steps = 960`
- recipe budget shape:
  - `1-3` baseline/exploration recipes
  - up to `4` exploitation recipes derived from the current incumbent
  - when new ROI-critical interventions land together, at most one augmentation branch and one
    KD/ranking branch may be added in the same invocation

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
