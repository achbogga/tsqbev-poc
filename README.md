# tsqbev-poc

`tsqbev-poc` is an evidence-driven multimodal 3D perception research repo focused on a
deployment-oriented student model, strong unrestricted teachers, and a local research-memory stack
that keeps the agenda grounded in exact artifacts plus primary-source literature.

[![CI](https://github.com/achbogga/tsqbev-poc/actions/workflows/ci.yml/badge.svg)](https://github.com/achbogga/tsqbev-poc/actions/workflows/ci.yml)
![Scale Up](https://img.shields.io/badge/scale_up-blocked-d73a49)
![Research Memory](https://img.shields.io/badge/research_memory-promoted_build_live-1f883d)
![Best Local mini_val NDS](https://img.shields.io/badge/best_mini__val_NDS-0.1833-1f883d)
![BEVFusion Detection](https://img.shields.io/badge/BEVFusion_det_nuScenes-0.7072_NDS-1f883d)

## Current Public Status

| Track | Current state | Evidence |
| --- | --- | --- |
| Best trusted local control | `NDS 0.1833`, `mAP 0.1814`, `18.34 ms`, `40.0` boxes/sample | [artifacts/research_v29_continuation_v1/research_loop/summary.json](artifacts/research_v29_continuation_v1/research_loop/summary.json) |
| Reproduced upstream teacher/control | `BEVFusion` on `nuScenes` val at `NDS 0.7072`, `mAP 0.6730` | [artifacts/bevfusion_repro/bevfusion_bbox_summary.json](artifacts/bevfusion_repro/bevfusion_bbox_summary.json) |
| Frontier camera branch | first real `DINOv3` probe is promising but still below the trusted local control | [artifacts/foundation_v3_dinov3_teacher_vits16_36ep_v1/epoch022_probe_r4/metrics/nuscenes/metrics_summary.json](artifacts/foundation_v3_dinov3_teacher_vits16_36ep_v1/epoch022_probe_r4/metrics/nuscenes/metrics_summary.json) |
| Lane branch | isolated `OpenLane V1` warm-start is viable; current naive joint detection+lane path is not trustworthy | [artifacts/openlane_v1_warmstart_v1/openlane_train.log](artifacts/openlane_v1_warmstart_v1/openlane_train.log), [artifacts/joint_public_v2_manual_eval/official_eval/epoch_031/nuscenes/metrics/metrics_summary.json](artifacts/joint_public_v2_manual_eval/official_eval/epoch_031/nuscenes/metrics/metrics_summary.json) |
| Knowledge base | `57` structured cards, `117` asset refs, `105` unique mirrored source assets | [artifacts/knowledge_assets/coverage_summary.json](artifacts/knowledge_assets/coverage_summary.json) |
| Memory layer | promoted exact build is live at repo SHA `647b888`, with `65` indexed facts and coherent current-build state | [artifacts/memory/sync_manifest.json](artifacts/memory/sync_manifest.json) |

## What This Repo Is

- a public research scaffold for multimodal 3D perception on public datasets
- a place to reproduce strong teacher ceilings, not just train a small student
- a knowledge-first repo: papers, official code, checkpoints, and our own artifacts are indexed
- a deployment-aware codebase with latency and export sanity treated as first-class outcomes

## Current Architecture Direction

The active reset stack is:

- deployable student: `Sparse4D`-style sparse temporal core
- camera foundation priors: `DINOv3` first, `DINOv2` fallback
- camera-to-BEV bridge: `BEVFormer v2`-style perspective supervision
- strong teachers and controls: `BEVFusion`, `OpenPCDet`, `CenterPoint-PointPillar`
- optional region priors: `SAM 2.1` as an offline teacher path
- lane/map branch: `MapTRv2`-style staged vector head, only after detection is stable

The detailed live agenda is in [docs/research-agenda.md](docs/research-agenda.md).

## Quick Start

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev --extra data
uv run pytest -q
uv run tsqbev smoke
```

If you have public datasets available locally:

```bash
uv run tsqbev check-data --dataset-root /path/to/dataset/root
```

## Research Memory And Knowledge Base

The repo ships with:

- exact experiment state in DuckDB
- semantic evidence indexing in Qdrant
- optional distilled memory in Mem0
- a local mirrored asset store for official papers, repos, and checkpoints

Useful commands:

```bash
uv run tsqbev memory-health
uv run tsqbev memory-backfill
uv run tsqbev research-report
uv run tsqbev knowledge-assets-status
uv run tsqbev knowledge-assets-sync
```

Important behavior:

- reranking is configured `Cohere-first` with local fallback
- if a hosted reranker credential is absent, the stack falls back to local reranking without
  blocking the repo
- promoted exact-memory builds are versioned under `.local/research-memory/builds/`

More detail:

- [docs/research-memory.md](docs/research-memory.md)
- [docs/research-assets.md](docs/research-assets.md)
- [docs/frontier-knowledge-base.md](docs/frontier-knowledge-base.md)

## Public Datasets And Baselines

- `nuScenes` and `OpenLane` are the public dataset targets
- `BEVFusion` is reproduced as the strong public teacher/control ceiling
- `OpenPCDet CenterPoint-PointPillar` is the verified external teacher bootstrap path

Start here:

- [docs/training-baselines.md](docs/training-baselines.md)
- [docs/teacher-bootstrap.md](docs/teacher-bootstrap.md)
- [docs/bevfusion-baseline-runbook.md](docs/bevfusion-baseline-runbook.md)
- [docs/lane-datasets.md](docs/lane-datasets.md)

## Research Agenda

The live evidence-backed agenda is in [docs/research-agenda.md](docs/research-agenda.md).

Current top-line agenda:

- hold the `v29` control as the only trusted local baseline
- continue only geometry-aware frontier branches that have a path to beating that control
- stop naive joint detection+lane coupling until official detection non-regression is enforced
- prioritize world-aligned `DINOv3 + perspective supervision + teacher BEV distillation`
- keep release/bootstrap quality high enough that external contributors can work from forks

## Contributing And Forking

- contributor workflow: [CONTRIBUTING.md](CONTRIBUTING.md)
- release/bootstrap checklist: [RELEASE.md](RELEASE.md)

The short version:

- keep changes scoped
- do not commit secrets
- do not make benchmark claims without linking the exact artifact
- keep generated report churn intentional

## Repo Layout

```text
docs/             architecture notes, agenda, runbooks, public docs
specs/            implementation contracts
src/tsqbev/       core code, training, evaluation, memory, KB tooling
tests/            unit and integration tests
research/         bounded experiment scripts and structured knowledge packs
artifacts/        generated run outputs, summaries, and mirrored-state manifests
```

## Source Grounding

This repo prefers primary sources:

- official papers
- official codebases
- official checkpoint pages
- exact local artifacts produced by this repo

The source map is maintained in [docs/reference-matrix.md](docs/reference-matrix.md).
