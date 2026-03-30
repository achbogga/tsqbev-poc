# tsqbev-poc

`tsqbev-poc` is a public proof-of-concept repository for a multimodal temporal sparse-query BEV stack built for open datasets and deployment validation.

[![CI](https://github.com/achbogga/tsqbev-poc/actions/workflows/ci.yml/badge.svg)](https://github.com/achbogga/tsqbev-poc/actions/workflows/ci.yml)
![Scale Up](https://img.shields.io/badge/scale_up-blocked-d73a49)
![Teacher Cache](https://img.shields.io/badge/teacher_cache-verified-1f883d)
![W%26B](https://img.shields.io/badge/W%26B-online-FFBE00)
![Best mini_val NDS](https://img.shields.io/badge/best_mini__val_NDS-0.0158-f2cc60)

- LiDAR grounds 3D object anchors and geometry
- cameras provide semantics, sparse refinement, and lane structure
- map priors are optional
- temporal state is sparse and streaming
- distillation is designed in from the start
- optional external LiDAR teacher bootstrap is now scaffolded
- ONNX and TensorRT deployment are first-class concerns

This repo is intentionally small and evidence-driven. Every substantive module is tied back to an original paper and, where available, an official codebase. Local generated summaries are treated as internal synthesis only. The repo cites the underlying original papers, official codebases, and our own public repo/paper artifacts instead.
When tracking is enabled, runs are mirrored to Weights & Biases under the entity `achbogga-track`.

## Quick Status

| Track | Status | Evidence |
| --- | --- | --- |
| CI | 🟢 Passing | `ruff`, `mypy`, `pytest` currently pass locally; GitHub Actions badge is wired in |
| Current mini incumbent | 🟡 Real but weak | `mini_propheavy_mbv3_frozen_query_boost`, `mini_val NDS = 0.01581`, `mAP = 1.11e-04`, `17.19 ms` |
| Teacher bootstrap | 🟢 Verified | external OpenPCDet `CenterPoint-PointPillar` reached `0.4997 NDS` on `mini_val`; cache coverage is full |
| Teacher lift into student | 🟡 Partial | corrected 32-sample teacher-anchor overfit control reached `NDS = 0.0484`, `mAP = 0.0150`; the teacher-KD path is still weak and paired `mini_val` lift is still unproven |
| Recovery branch | 🟡 Code-ready | best-checkpoint evaluation, focal hard negatives, bounded calibration, anchor-first teacher routing, explicit teacher-anchor vs teacher-KD separation, and class-balanced teacher seed selection are implemented and tested |
| Scale-up readiness | 🔴 Blocked | overfit, repeatability, and mini generalization gates are still failing |
| Tracking | 🟢 Online | W&B smoke run synced under `achbogga-track` |

The current state is straightforward: the public repo is healthy, tested, deploy-checked, and tracked, but the student model is not yet strong enough to justify scaling compute by 10x.

## What This Repo Is

- a minimal multimodal BEV research artifact
- a spec-driven and test-first implementation
- a public `nuScenes` / `OpenLane` / `MapTR`-style prototype
- a deployment-oriented codebase with measured RTX 5000 latency

## What This Repo Is Not

- a large-scale training platform
- a private or proprietary dataset integration layer
- an unbounded autonomous research loop
- a finished embedded deployment product

## Architecture At A Glance

```mermaid
flowchart LR
    A[Multi-view images] --> B[Image backbone + neck]
    C[LiDAR points] --> D[Pillar encoder]
    B --> E[2D proposal head]
    E --> F[Proposal-ray seeds]
    G[Learned global seeds] --> H[Tri-source router]
    D --> H
    F --> H
    B --> I[Sparse cross-view sampler]
    H --> I
    I --> J[Query fusion]
    K[Temporal state t-1] --> L[Temporal updater]
    J --> L
    L --> M[Object head]
    L --> N[Lane head]
    O[Optional map priors] --> N
    L --> P[Temporal state t]
```

More detail and additional diagrams are in [docs/architecture.md](docs/architecture.md) and the paper in [docs/paper/tsqbev_short_paper.pdf](docs/paper/tsqbev_short_paper.pdf).

## Current Public Scope

- Object detection: `nuScenes`, with `v1.0-mini` as the active local research contract
- Lane supervision: `OpenLane V1`
- Map priors: `MapTR`-style vectorized public priors
- Teacher bootstrap: optional cached external LiDAR teacher path, starting with public `CenterPoint-PointPillar` style teachers
- Deployment validation: ONNX export and TensorRT engine build for the exportable core
- Experiment tracking: optional W&B logging for baselines, gates, and research-loop recipes

## Measured Results

RTX 5000 latency, batch size `1`, image size `256x704`:

| Path | Mean ms | p95 ms |
| --- | ---: | ---: |
| Full model, eager PyTorch | 10.872 | 10.977 |
| Exportable core, PyTorch FP32 | 7.883 | 8.057 |
| Exportable core, PyTorch FP16 | 7.492 | 7.650 |
| Exportable core, TensorRT FP16-enabled engine | 0.785 | 0.795 |

The latency measurements are summarized in [docs/benchmarks/rtx5000.md](docs/benchmarks/rtx5000.md). The TensorRT result applies to the current exportable core only, not the full end-to-end multimodal pipeline.

Latest completed teacher-backed bounded `nuScenes v1.0-mini` sweep:

| Run | Stage | Key Setting | Val Total | mAP | NDS | Mean ms | Source Mix | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Balanced `MobileNetV3-Large` | baseline | frozen, `q_lidar=96`, `q_2d=64` | 21.0663 | `8.7234e-05` | `4.3617e-05` | 17.2304 | `50/33/17` | discard |
| Proposal-heavy `MobileNetV3-Large` | explore | frozen, `q_lidar=64`, `q_2d=96` | 24.7025 | `2.6575e-04` | `1.3288e-04` | 17.2103 | `33/50/17` | discard |
| Proposal-heavy `EfficientNet-B0` | explore | frozen, `q_lidar=64`, `q_2d=96` | 19.4404 | `4.6209e-05` | `0.01247` | 21.9496 | `33/50/17` | promote |
| Teacher-seeded `EfficientNet-B0` | exploit | frozen, `replace_lidar` teacher seeds | 22.9615 | 0.0 | 0.0 | 21.1112 | `33/50/17` | discard |
| Query-boost `EfficientNet-B0` | exploit | frozen, `q_2d=112`, `max_q=112` | 19.6120 | `5.2059e-05` | `0.01165` | 21.7800 | `31/54/15` | discard |

The loop selects by official `mini_val` `NDS`, then `mAP`, then loss. The current incumbent is the
proposal-heavy frozen `EfficientNet-B0` recipe. It is a real, reproducible local signal, but still
well below the threshold required to justify scaling the training budget. The sweep artifacts are
summarized in [docs/benchmarks/nuscenes-mini.md](docs/benchmarks/nuscenes-mini.md) and the latest
teacher-backed bounded run in `artifacts/research_teacher_v1/research_loop/`.
The `replace_lidar` teacher row above should no longer be treated as valid evidence on the current
branch: batching used to drop `teacher_targets`, so teacher lift is being rerun with the fixed
collator, geometry-aware teacher class and box supervision, and the stronger `replace_lidar`
mode.
The same loop is also tracked in W&B when credentials are available; hyperparameter and
performance sweeps stay grouped within the same architecture-family project, while materially
different architectures are logged to separate projects.

The current scale-up answer is still "not yet": the repaired 32-sample overfit gate still failed,
with `train_ratio=0.5310`, same-subset `NDS=0.0085868`, and same-subset `mAP=0.0005329`. See
[docs/scaling-gates.md](docs/scaling-gates.md).

The strongest measured overfit rescue so far is the corrected teacher-seeded variant on the same
32-sample subset: `NDS=0.0401`, `mAP=0.0214`, and `16.92 ms`, but it still failed the gate because
`car AP @ 4.0m` remained `0.0` and the subset did not overfit enough. The current branch therefore
pushes ROI into ranking/overproduction fixes first: selected-checkpoint evaluation, focal-style
hard-negative training, and bounded export calibration.

The current first-principles diagnosis is narrower now: the fixed 32-sample subset contains real
cars, and the teacher cache contains many more car anchors than the student is emitting, but raw
teacher-score truncation was still overfilling the fixed teacher seed budget with barrier and
pedestrian anchors. Teacher-anchor runs now default to a class-balanced teacher seed selection
policy before any further KD complexity is added.

External teacher bootstrap is now verified separately:

| Teacher | Split | External mAP | External NDS | Cache Coverage |
| --- | --- | ---: | ---: | ---: |
| OpenPCDet `CenterPoint-PointPillar` | `nuScenes v1.0-mini` `mini_val` | 0.4369 | 0.4997 | `323 / 323` train, `81 / 81` val |

That benchmark and the audited cache import are documented in
[docs/benchmarks/openpcdet-centerpoint-mini.md](docs/benchmarks/openpcdet-centerpoint-mini.md).

## Source Grounding

Primary references include:

- [DETR3D](https://proceedings.mlr.press/v164/wang22b.html)
- [PETR / PETRv2](https://github.com/megvii-research/PETR)
- [StreamPETR](https://github.com/exiawsh/StreamPETR)
- [Sparse4D](https://github.com/HorizonRobotics/Sparse4D)
- [SparseBEV](https://github.com/MCG-NJU/SparseBEV)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [BEVDistill](https://arxiv.org/abs/2211.09386)
- [PillarNet](https://github.com/VISION-SJTU/PillarNet)
- [CMT](https://github.com/junjie18/CMT)
- [BEVFusion](https://arxiv.org/abs/2205.13542)
- [MapTR](https://github.com/hustvl/MapTR)
- [HotBEV](https://proceedings.neurips.cc/paper_files/paper/2023/file/081b08068e4733ae3e7ad019fe8d172f-Paper-Conference.pdf)

The full source map is in [docs/reference-matrix.md](docs/reference-matrix.md).

## Docs

- [Architecture](docs/architecture.md)
- [RTX 5000 latency benchmark](docs/benchmarks/rtx5000.md)
- [nuScenes mini baseline](docs/benchmarks/nuscenes-mini.md)
- [OpenPCDet teacher bootstrap benchmark](docs/benchmarks/openpcdet-centerpoint-mini.md)
- [nuScenes overfit gate](docs/benchmarks/nuscenes-overfit-gate.md)
- [Scaling gates](docs/scaling-gates.md)
- [Teacher bootstrap](docs/teacher-bootstrap.md)
- [OpenPCDet CenterPoint teacher runbook](docs/openpcdet-centerpoint-teacher.md)
- [Reference matrix](docs/reference-matrix.md)
- [Public baseline workflow](docs/training-baselines.md)
- [Implementation plan](docs/plan.md)
- [Short paper PDF](docs/paper/tsqbev_short_paper.pdf)
- [Short paper LaTeX](docs/paper/tsqbev_short_paper.tex)

## Repo Layout

```text
docs/           plan and evidence trail
specs/          implementation contracts
src/tsqbev/     minimal multimodal implementation
tests/          isolated and integration tests
research/       bounded research-loop recipes and evidence
artifacts/      local run outputs, exports, and mini-baseline results
```

## Quick Start

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
uv run pytest
uv run tsqbev smoke
uv run tsqbev train-step
uv run tsqbev eval
uv run tsqbev bench
```

For real public-dataset baselines:

```bash
uv sync --extra dev --extra data
uv run tsqbev check-data --dataset-root /path/to/dataset/root
```

The full workflow for `nuScenes` and `OpenLane` is documented in [docs/training-baselines.md](docs/training-baselines.md). Full `v1.0-trainval` accuracy is not published yet; the repo currently reports measured `v1.0-mini` results only.

The repo now also has a dedicated tiny-subset overfit gate for `nuScenes`, which evaluates the
exact same fixed token subset used for training:

```bash
uv run tsqbev overfit-nuscenes \
  --dataset-root /path/to/nuscenes \
  --artifact-dir artifacts/gates \
  --preset rtx5000-nuscenes-query-boost \
  --version v1.0-mini \
  --train-split mini_train \
  --subset-size 32 \
  --epochs 128 \
  --max-train-steps 1024 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --optimizer-schedule constant \
  --grad-clip-norm 5.0 \
  --loss-mode focal_hardneg \
  --score-threshold-candidates 0.05 0.15 0.25 \
  --top-k-candidates 32 64 112 \
  --device cuda
```

The active bounded research loop is currently scoped to `nuScenes v1.0-mini` only and now follows
the strongest transferable ideas from Karpathy's `autoresearch`: one incumbent, a fixed comparable
train-step budget, bounded exploration, bounded exploitation, append-only ledgers, and explicit
promote/discard semantics.

```bash
uv run tsqbev research-loop \
  --dataset-root /path/to/nuscenes \
  --artifact-dir artifacts/baselines \
  --max-experiments 5 \
  --device cuda
```

Teacher-assisted training is now scaffolded through cached teacher targets:

```bash
uv run tsqbev cache-teacher-nuscenes \
  --dataset-root /path/to/nuscenes \
  --version v1.0-mini \
  --result-json /path/to/external_teacher_results.json \
  --teacher-cache-dir /path/to/teacher-cache
```

Then train from the cache:

```bash
uv run tsqbev train-nuscenes \
  --dataset-root /path/to/nuscenes \
  --preset rtx5000-nuscenes-teacher \
  --version v1.0-mini \
  --train-split mini_train \
  --split mini_val \
  --teacher-kind cache \
  --teacher-cache-dir /path/to/teacher-cache
```

Use `--preset rtx5000-nuscenes-teacher` or `--teacher-seed-mode replace_lidar` when the
cached teacher outputs include `object_boxes`, `object_labels`, and `object_scores`. On the
current branch, `replace_lidar` is the default full LiDAR-seed replacement mode and
`replace_lidar_refs` is the legacy ref-only ablation.

Audit cache coverage before claiming any teacher lift:

```bash
uv run tsqbev audit-teacher-cache-nuscenes \
  --dataset-root /path/to/nuscenes \
  --version v1.0-mini \
  --split mini_val \
  --teacher-cache-dir /path/to/teacher-cache \
  --output-dir artifacts/teacher_cache_audit
```

The exact external OpenPCDet `CenterPoint-PointPillar` runbook is documented in
[docs/openpcdet-centerpoint-teacher.md](docs/openpcdet-centerpoint-teacher.md).

Check whether a machine is actually ready for that external teacher path:

```bash
uv run tsqbev check-openpcdet-env \
  --openpcdet-repo-root /path/to/OpenPCDet
```

For CUDA deployment validation on supported NVIDIA systems:

```bash
uv sync --extra dev --extra deploy
uv run tsqbev trt-bench
```

## Validation Status

- `ruff` clean
- `mypy` clean
- `pytest` passing
- ONNX export smoke passing
- TensorRT engine build validated on RTX 5000
- optional W&B tracking integrated for all substantive experiment entrypoints
- latest bounded `nuScenes v1.0-mini` teacher-backed sweep recorded with official per-recipe `mini_val` evaluation
- current best completed `mini_val` result: `NDS = 0.01247`, `mAP = 4.6209e-05`, `mean latency = 21.95 ms`
- explicit scale gates added; current answer remains "do not scale by 10x compute yet"
- exact-token overfit gate recorded; current verdict is fail
- optional external LiDAR teacher cache/provider scaffolding added and tested
- external OpenPCDet `CenterPoint-PointPillar` teacher verified at `0.4997 NDS` on `mini_val`
- teacher-seed replacement ablation recorded; current verdict is regression and needs redesign
- bounded mini-dataset research loop enabled via `program.md`
- research loop upgraded to staged baseline/explore/exploit with `results.tsv`, per-run `manifest.json`, and machine-readable `scale_gate_verdict`
