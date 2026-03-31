# tsqbev-poc Plan

## Purpose

`tsqbev-poc` is a public-facing proof-of-concept repo for a production-minded multimodal BEV
system.

The repo is meant to:

- validate the architecture quickly
- stay minimal and elegant
- be spec-driven and test-first
- remain clean enough for paper submission and public review
- avoid premature scale infrastructure

The repo is not a final large-scale production training system.
It is a focused open-source POC for public datasets and deployable research artifacts.
The current codebase is now being migrated from the legacy sparse-query line toward a dense BEV
fusion reset stack built from public upstreams.

## Reset Target Stack

The new primary target is a dense-BEV fusion stack with public upstream support:

- LiDAR BEV encoder: `OpenPCDet` / `CenterPoint-PointPillar`
- camera BEV encoder: `BEVDet` / `BEVDepth`
- fusion trunk: `BEVFusion`
- detection head: `CenterHead`-style dense detection
- lane / map head: `MapTRv2`-style vector output on the shared BEV
- efficiency branch: `EfficientViT`, then `OFA` / `AMC` / `HAQ`
- optional dense priors: `DINOv2` / `DINOv3`

The legacy sparse-query line remains in this repo as comparison evidence and a fallback benchmark,
but it is no longer the primary architecture target.

## Standing Research Directives

These are durable repo-level instructions:

- review every important design decision from first principles before extending it
- use primary papers, official repos, and official weights for unstable SOTA areas
- treat KD as a large design space, not a single tactic
- prefer the smallest high-ROI experiment that can falsify the current bottleneck hypothesis
- keep moving after each bounded result until the next step is no longer clearly justified
- stop and reassess when the current branch turns into a rabbit hole

The repo now also carries a lightweight `token_burn_score` rule in `AGENTS.md`, `program.md`, and
`specs/004-research-loop-contract.md` so each direction is explicitly scored on expected ROI,
integration cost, uncertainty, and evidence gain before more time or compute is spent.

## Current Status

- The public repo is bootstrapped, tested, and published on GitHub.
- The public `nuScenes` data is present under `/mnt/storage/research/nuscenes`.
- Real-data `nuScenes` readiness checks pass.
- All selected dense-BEV upstream repos are now cloned locally under `/home/achbogga/projects`
  with branch and head-sha provenance tracked in the repo.
- The official BEVFusion Docker workflow is now codified in this repo with a dedicated readiness
  checker, runbook, and helper scripts, because the host Python version is outside BEVFusion's
  official runtime range.
- The current BEVFusion reproduction blocker is now explicit and evidence-backed: the local
  `nuScenes` dataset root is still missing the official map-expansion bundle under
  `maps/{basemap,expansion,prediction}`, and BEVFusion's shared nuScenes pipeline constructs
  `NuScenesMap` even for the detection config.
- A float32 real-data smoke run on the RTX 5000 completed successfully on a tiny subset, which confirmed the loader and training path are functional on real data.
- The repo is now in migration state: the legacy sparse-query line remains documented and measured,
  but the primary target is a public dense-BEV fusion stack.
- All primary upstream reset repos are now cloned locally with recorded branch and `HEAD`
  provenance.
- The official `BEVFusion` path now has a repo-native readiness check, a dedicated runbook, and a
  repo-local ann-file helper that emits the exact `nuscenes_infos_train.pkl` / `nuscenes_infos_val.pkl`
  files expected by the upstream configs without editing the archived upstream source tree.
- The active local research contract is now bounded `nuScenes v1.0-mini`, not full `v1.0-trainval`.
- The first bounded `v1.0-mini` sweep established a functioning public baseline but remained near-zero on official metrics.
- The strengthened bounded `v1.0-mini` sweep fixed the routed query-bank collapse and changed recipe selection to official `mini_val` `NDS`, then `mAP`, then loss.
- The current best public mini result is `mini_propheavy_mbv3_frozen_query_boost` with
  `val total = 20.1352`, `mAP = 1.1140e-04`, and `NDS = 0.0158068933`.
- The current best source mix is `31.25% LiDAR / 53.57% proposal / 15.18% global`, which
  preserves the stable non-collapsed multimodal routing regime while improving official metrics.
- The repo now explicitly blocks 10x compute scale-up until the gates in `specs/005-scale-gate-contract.md` are cleared.
- W&B tracking is wired into the experiment entrypoints and stays advisory only; tracking failures do not change training or selection outcomes.
- Optional external LiDAR teacher scaffolding is now present through typed cache/provider contracts and a dataset wrapper, with `CenterPoint-PointPillar` as the first target backend.
- The repo now accepts standard nuScenes detection JSON from an external teacher and converts it
  into repo-local teacher-cache records.
- The repo now has an explicit teacher-cache audit path for `mini_train` / `mini_val` coverage
  before any teacher-lift claim is allowed.
- The external OpenPCDet teacher path is now verified locally on this workstation:
  CUDA extensions build, the official pretrained `CenterPoint-PointPillar` checkpoint runs on
  `v1.0-mini`, and the resulting teacher cache covers `323 / 323` `mini_train` samples and
  `81 / 81` `mini_val` samples.
- The earlier teacher-lift artifact was invalidated after a batched-collation bug was found to
  drop `teacher_targets` from real dataset batches.
- The corrected teacher-backed path now carries teacher targets through batching, uses
  geometry-aware teacher class and box supervision, and requires a same-invocation paired
  teacher-off / teacher-KD / teacher-seed comparison before any lift claim is accepted.
- The current branch now separates teacher-anchor and teacher-KD roles explicitly:
  teacher-anchor runs use full `replace_lidar` seed replacement with anchor-first routing,
  while teacher-KD remains a separate paired supervision path.
- The detection path now includes bounded center refinement around query refs, explicit objectness
  ranking, and exported-prediction geometry diagnostics in the research loop.
- The bounded research loop now measures exported-prediction geometry in the ego frame, not raw
  nuScenes global coordinates, and hard-blocks promotion when boxes are too numerous or too far
  away in ego range even if surrogate losses or official metrics improve.
- The bounded mini loop now mirrors the strongest transferable `autoresearch` mechanics more closely:
  incumbent-first execution, bounded exploration then exploitation, append-only `results.jsonl` and
  `results.tsv`, per-run `manifest.json`, a fixed comparable `max_train_steps=960` budget per
  recipe, explicit `promote/discard/crash` semantics, and a machine-readable `scale_gate_verdict`.
- The repo now also contains a dedicated `nuScenes` overfit-gate runner that trains and evaluates
  on the exact same fixed token subset through the official metric stack.
- The repaired overfit-gate artifact still failed, with `train_total_ratio = 0.5310`,
  same-subset `NDS = 0.0085868`, same-subset `mAP = 0.0005329`, `3` nonzero classes, and still no
  nonzero `car AP @ 4.0m`.
- The strongest measured recovery probe so far is the corrected balanced teacher-anchor overfit
  run on the same 32-sample subset, which reached `NDS = 0.1001`, `mAP = 0.1391`,
  `car AP @ 4.0m = 0.5327`, `7` nonzero classes, and `17.92 ms`.
- That repaired run still failed the scale gate, but now for a much narrower reason:
  `train_total_ratio = 0.4703` remained above the required `0.40` threshold even though the
  official metric thresholds and `car AP @ 4.0m` were satisfied.
- Direct artifact inspection now shows the fixed 32-sample subset contains `308` ground-truth
  cars and the external teacher cache contains many more car anchors than that, but the old raw
  teacher-score top-k truncation was still overfilling the fixed teacher seed budget with barrier
  and pedestrian anchors before student ranking/classification ever ran.
- The next first-principles recovery step is now implemented in code: anchor-first teacher-seed
  routing plus explicit disabling of extra KD during teacher-anchor runs, because the same teacher
  should not be asked to serve as both the anchor set and the logits target in a single recovery
  experiment.
- The current branch now implements the next ROI recovery slice:
  selected-checkpoint evaluation, focal-style hard-negative detection loss, bounded
  score-threshold / `top_k` calibration, and teacher-seed-first exploitation in the bounded loop.
- The current teacher-anchor branch now also uses a bounded class-balanced teacher seed-selection
  policy, which materially improved class diversity and unlocked the first overfit run above
  `0.10 NDS`.
- The next active diagnostic is the repaired overfit recovery loop on the same fixed 32-sample
  subset, not another unconstrained `mini_val` sweep.
- OpenLane support still needs version alignment against the OpenLane-V2 getting-started instructions before any lane baseline can be treated as final.
- The bounded mini-dataset research loop is now authorized via `program.md`.

## Milestones Achieved

- Created the clean `uv`-based repo and public docs.
- Removed private-dataset references from the public-facing plan and docs.
- Implemented typed multimodal contracts, dataset loaders, loss functions, export helpers, and training loops.
- Added unit, integration, and checkpoint round-trip tests.
- Verified synthetic forward/backward, export smoke, and repo lint/type checks.
- Verified the public `nuScenes` dataset root and executed real-data smoke training successfully in float32.
- Added a pretrained MobileNetV3 image-backbone path and batched training.
- Enabled a bounded `nuScenes v1.0-mini` research loop contract.
- Added explicit scale-gate and external-teacher bootstrap contracts.
- Added optional external teacher cache/provider scaffolding for pretrained LiDAR teachers.
- Added optional external teacher cache/provider scaffolding for LiDAR-strong distillation experiments.
- Added standard nuScenes teacher-result JSON ingestion and teacher-cache coverage auditing.
- Added an OpenPCDet CenterPoint-PointPillar runbook for the first external teacher bootstrap path.
- Upgraded the bounded local loop to a staged `baseline -> explore -> exploit` workflow aligned
  with the public `autoresearch` design intent while remaining bounded and ML-specific.
- Added per-run manifests, a human-readable TSV ledger, and explicit scale-gate verdict emission.
- Added optional W&B tracking with stable project grouping by architecture family.
- Added an exact OpenPCDet `CenterPoint-PointPillar` teacher runbook, grounded in the official
  config and export paths, without adding heavy runtime dependencies to the core repo.
- Verified the external OpenPCDet `CenterPoint-PointPillar` teacher end to end on
  `nuScenes v1.0-mini`.
- Verified repo-local teacher-cache conversion and full `mini_train` / `mini_val` coverage auditing.
- Cloned and provenance-checked the public dense-BEV reset upstream repos:
  `BEVFusion`, `BEVDet`, `MapTRv2`, `PersFormer`, `DINOv2`, `DINOv3`, and `EfficientViT`.
- Added a BEVFusion-specific official Docker runbook and helper scripts so the next reset milestone
  can be executed reproducibly on this workstation.

## Evidence Basis

- The public primary sources used by this repo are indexed in `docs/reference-matrix.md`.
- Local generated design summaries are treated as internal synthesis only; the repo cites the underlying original sources instead.
- The workflow staging is influenced by Andrej Karpathy's public `autoresearch` repository: <https://github.com/karpathy/autoresearch>

## Hard Constraints

- Do not run an unbounded autonomous `autoresearch` loop.
- Do not optimize for large-scale distributed training in v1.
- Keep the implementation dependency-light and pure PyTorch at the core.
- Make the repo functional first: setup, tests, forward pass, export smoke, perf harness, adapters.
- Treat GPU and Orin validation as separate acceptance stages after local functionality is stable.
- Keep the active loop limited to `nuScenes v1.0-mini`.

## Current Operating Assumption

- The GPU runtime is available and verified on the RTX 5000.
- Public-data work should stay measured and reproducible.
- Local tuning should use `v1.0-mini` first and only promote after recorded evidence.

## Legacy System Goal

The historical implementation goal was to build a multimodal temporal sparse-query detector with:

- LiDAR grounding for 3D object depth and anchors
- camera semantics for object refinement and lane reasoning
- temporal state for streaming stability
- optional HD map priors
- distillation designed in from day one
- graceful fallback behavior when LiDAR is degraded or absent

## Legacy Model Design

### Modalities

- `LiDAR`: primary grounding signal for final 3D object localization
- `Camera`: primary semantic signal and primary lane signal
- `HD Map Priors`: optional structured prior stream, especially for lane/map reasoning

### Query Initialization

Use a learnable tri-source query bank:

- `Q_lidar`: sparse queries seeded from LiDAR points via a lightweight pillar/BEV-lite encoder
- `Q_2d`: queries seeded from 2D camera proposals and backprojected along calibrated rays
- `Q_global`: a small learned global anchor bank for recall and unmatched cases

Concatenate all three and pass them through a small router that:

- adds source embeddings
- uses source confidence priors
- scores and filters candidates
- keeps the strongest queries for the multimodal decoder

This avoids the failure mode of LiDAR-only seeding while preserving the grounding benefit of LiDAR.

### Core Decoder

Default v1 architecture:

- lightweight image backbone
- 2-scale image neck
- lightweight pillar LiDAR encoder
- tri-source query router
- sparse calibrated camera sampler
- lightweight multimodal query fusion blocks
- persistent temporal state
- 3D object detection head
- camera-dominant lane head with LiDAR ground hints
- optional map-token fusion path

### Temporal State

Default temporal state:

- persistent sparse object queries
- optional sparse lane memory

Do not use dense temporal BEV caches in v1.

## Public Data Strategy

### Public Benchmarks

- `nuScenes`: object detection and public map-prior path
- `OpenLane v1`: lane supervision
- `nuScenes map expansion + MapTR-style vector priors`: first public map adapter

### Open Dataset Scope

Use only open-source dataset adapters in the public repo:

- `nuScenes` for object detection
- `OpenLane v1` for lane supervision
- `nuScenes` map expansion with MapTR-style vector priors for public map tokens

Do not include private dataset compatibility layers in this repository.

## Distillation Strategy

Distillation is part of v1 design, but not full-scale teacher execution.

The repo should include:

- teacher target interfaces
- cache readers/writers
- student distillation losses
- mock teacher fixtures for tests

The repo should not initially include:

- internal teacher training code
- cloud orchestration
- large-scale distillation runs

Primary distillation goals:

- transfer geometry priors into the smaller multimodal student
- stabilize training
- preserve a weaker but valid camera-fallback path under LiDAR dropout

## Performance Contract

### Single-Orin Targets

- production gate: `p95 <= 100 ms` on 1 Orin in FP16 TensorRT
- stretch gate: `p95 <= 50 ms` on 1 Orin in INT8 TensorRT

### Model Size Tiers

- small: `<= 30M params`
- default: `<= 55M params`
- large: `<= 85M params`

The default tier is the main POC target.

### Latency Prediction

The repo should include a simple predictor for pre-filtering configurations:

`pred_ms = b0 + b1*params_M + b2*sample_ops_M + b3*lidar_pillars_K + b4*T + b5*activations_MB`

Where:

- `sample_ops_M` depends on `V * L * (M_obj + M_lane) * K`
- `lidar_pillars_K` is active non-empty pillar count
- `T` is temporal window usage

This predictor is only a gate and triage tool.
Real latency decisions come from measured microbenchmarks and end-to-end timings.

## Functional Milestone Before Bounded Research

The repo is considered ready only when all of the following are true:

- environment setup works
- `uv sync` succeeds
- unit tests pass
- synthetic multimodal forward pass works
- backward pass works
- ONNX export smoke passes
- perf microbenches run
- manual train/eval entrypoints run
- the bounded mini-dataset research loop is explicitly authorized in `program.md`

No bounded loop should exist in executable form before these gates are green.

## Implementation Sequence

### Phase 0: Post-Reboot Verification

After reboot:

1. verify `uv`
2. verify `python3`
3. verify `nvidia-smi`
4. decide whether GPU validation is possible or CPU/synthetic-only for now

### Phase 1: Minimal Repo Bootstrap

Create only the essential files:

- `Apache-2.0`
- `README.md`
- `pyproject.toml`
- `.gitignore`
- `program.md` with explicit bounded-loop status
- initial `specs/`

No experiment loop logic in this phase.

### Phase 2: Contracts First

Implement typed contracts for:

- `SceneBatch`
- `QuerySeedBank`
- `MapPriorBatch`
- `TeacherTargets`
- config schemas

Write tests for these before model code.

### Phase 3: Geometry And Seed Paths

Implement and test:

- calibration and projection utilities
- lightweight pillarization
- LiDAR query seeds
- 2D proposal-ray query seeds
- learned global seeds
- tri-source query router

### Phase 4: Core Model Skeleton

Implement and test:

- image backbone wrapper
- 2-scale neck
- sparse camera sampler
- multimodal query fusion block
- temporal state update
- OD head
- lane head
- optional map token path

### Phase 5: Adapters

Add:

- `nuScenes` OD adapter
- `OpenLane` lane adapter
- `nuScenes` map-prior adapter

All adapters require contract tests with synthetic fixtures.

### Phase 6: Harnesses

Add:

- manual train entrypoint
- manual eval entrypoint
- ONNX export smoke
- latency predictor
- microbench entrypoints

### Phase 7: Distillation Interfaces

Add:

- teacher target schema
- distillation losses
- cached teacher target loading
- mock teacher fixtures

Still no full cloud training workflow.

### Phase 8: Controlled Research Scaffolding

After the repo became functional:

- add `program.md`
- add bounded experiment bookkeeping
- add append-only result logging scaffolding

This phase enables a bounded `nuScenes v1.0-mini` loop only.

## Test Plan

### Unit Tests

Write isolated tests for:

- geometry and projection
- pillarization
- 2D query backprojection
- query router behavior
- sparse sampling
- temporal state update
- OD head
- lane head
- map-token fusion
- distillation losses
- config validation

### Property Tests

Use property-style testing for:

- projection invariants
- query source routing invariants
- temporal alignment invariants
- lane polyline normalization
- map coordinate transforms

### Integration Tests

Add synthetic integration tests for:

- multimodal forward pass
- backward pass
- OD-only batch
- lane-only batch
- map-prior batch
- LiDAR-dropout fallback
- ONNX export smoke

### Adapter Tests

Add adapter contract tests for:

- `nuScenes`
- `OpenLane`
- map priors

## Research Loop Policy

The bounded research loop is now enabled with the following constraints:

- mutate only bounded research surfaces
- never rewrite core contracts silently
- log results append-only
- remain human-overridable
- stay on `nuScenes v1.0-mini`
- avoid unbounded or recursive execution

## Deliverable Definition For V1

V1 is complete when the repo has:

- a clear plan and specs
- a clean install path
- tested core contracts
- a working multimodal model skeleton
- tested adapters
- export smoke coverage
- latency harnesses
- distillation interfaces
- a bounded research scaffold

V1 still does not require:

- distributed training
- full teacher training
- Orin-perfect optimization
- unbounded autonomous experimentation

## Immediate Next Step

Resume with:

1. publish measured `v1.0-mini` results into the docs and paper
2. push the bounded-loop and mini-baseline updates to the public repo
3. decide whether to keep iterating on `v1.0-mini` or promote to `v1.0-trainval` later
4. decide whether the lane path should target OpenLane V1 or be adapted to OpenLane-V2 before any lane baseline
