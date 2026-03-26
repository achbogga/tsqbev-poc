# tsqbev-poc Plan

## Purpose

`tsqbev-poc` is a public-facing proof-of-concept repo for a production-minded multimodal temporal sparse-query BEV system.

The repo is meant to:

- validate the architecture quickly
- stay minimal and elegant
- be spec-driven and test-first
- remain clean enough for paper submission and public review
- avoid premature scale infrastructure

The repo is not the final large-scale Torc training system.
It is a focused POC that can later be integrated into `torc_ml`.

## Evidence Basis

- The public primary sources used by this repo are indexed in `docs/reference-matrix.md`.
- The local summary PDF in `/home/achbogga/projects/Production-ready Temporal Sparse Query BEV for Torc on NVIDIA Orin.pdf` is treated as internal synthesis only; the repo cites the underlying original sources instead.
- The workflow staging is influenced by Andrej Karpathy's public `autoresearch` repository, but the autonomous loop remains disabled during bootstrap: <https://github.com/karpathy/autoresearch>

## Hard Constraints

- Do not enable or run any autonomous `autoresearch` loop yet.
- Do not optimize for large-scale distributed training in v1.
- Keep the implementation dependency-light and pure PyTorch at the core.
- Make the repo functional first: setup, tests, forward pass, export smoke, perf harness, adapters.
- Treat GPU and Orin validation as separate acceptance stages after local functionality is stable.

## Current Operating Assumption

- The machine is mid-driver-update / restart cycle.
- Real GPU verification may still be broken immediately after reboot.
- Implementation should proceed CPU-first and synthetic-data-first if GPU runtime is still unhealthy.

## System Goal

Build a multimodal temporal sparse-query detector with:

- LiDAR grounding for 3D object depth and anchors
- camera semantics for object refinement and lane reasoning
- temporal state for streaming stability
- optional HD map priors
- distillation designed in from day one
- graceful fallback behavior when LiDAR is degraded or absent

## Primary Model Design

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

### Torc Compatibility

Use a thin Torc compatibility adapter informed by:

- `torc_ml/projects/scene_modeling/scene_modeling/data/datasets`

Do not vendor Torc code into the public repo.
Mirror schema and calibration contracts only where needed.

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

## Functional Milestone Before Any Research Loop

The repo is considered ready only when all of the following are true:

- environment setup works
- `uv sync` succeeds
- unit tests pass
- synthetic multimodal forward pass works
- backward pass works
- ONNX export smoke passes
- perf microbenches run
- manual train/eval entrypoints run
- auto-research remains disabled

No autonomous loop should exist in executable form before these gates are green.

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
- `program.md` with explicit disabled status
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
- thin Torc adapter

All adapters require contract tests with synthetic fixtures.

### Phase 6: Harnesses

Add:

- manual train entrypoint
- manual eval entrypoint
- ONNX export smoke
- latency predictor
- microbench entrypoints

Still no autonomous research loop.

### Phase 7: Distillation Interfaces

Add:

- teacher target schema
- distillation losses
- cached teacher target loading
- mock teacher fixtures

Still no full cloud training workflow.

### Phase 8: Controlled Research Scaffolding

Only after the repo is functional:

- add `program.md`
- add disabled experiment bookkeeping
- add append-only result logging scaffolding

This phase does not enable any always-on autonomous loop.

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
- Torc thin adapter

## Research Loop Policy

The research loop is explicitly disabled until:

- local functionality is green
- the user confirms budget is available
- the user explicitly authorizes enabling it

When that phase comes later, the loop must:

- mutate only bounded research surfaces
- never rewrite core contracts silently
- log results append-only
- remain human-overridable

But none of that should be implemented in executable form yet.

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
- a disabled research scaffold

V1 does not require:

- distributed training
- full teacher training
- Orin-perfect optimization
- autonomous experimentation

## Immediate Next Step After Reboot

Resume with:

1. post-reboot environment verification
2. repo bootstrap files
3. specs and tests
4. core implementation

No autonomous loop work until explicitly approved later.
