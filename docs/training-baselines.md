# Public Baseline Workflow

This repository now contains real-data training, evaluation, and a bounded local research loop for public baselines. The active local contract is `nuScenes v1.0-mini`, not the full `v1.0-trainval` split.

Grounding for the acquisition requirements:

- `nuScenes` download and devkit structure: <https://github.com/nutonomy/nuscenes-devkit>
- `OpenLane V1` dataset structure and evaluation kit: <https://github.com/OpenDriveLab/OpenLane>

## Required Public Data

### nuScenes

The code expects a standard `nuScenes` dataroot containing at least:

```text
<nuscenes-root>/
  v1.0-mini/
    scene.json
    sample.json
    sample_data.json
    calibrated_sensor.json
    ego_pose.json
    sample_annotation.json
  v1.0-trainval/
    scene.json
    sample.json
    sample_data.json
    calibrated_sensor.json
    ego_pose.json
    sample_annotation.json
  samples/
    CAM_FRONT/
    LIDAR_TOP/
  sweeps/
  maps/
```

The official devkit supports both the full train/val split and the mini split. This repo’s active research loop uses `mini_train` / `mini_val`.

### OpenLane V1

The code expects the official folder structure documented by OpenLane:

```text
<openlane-root>/
  images/
    training/
    validation/
  lane3d_300/
    training/
    validation/
  lane3d_1000/
    training/
    validation/
```

OpenLane V1 is distributed via the project’s official form and is built on top of Waymo data, so the upstream Waymo terms also apply.

## Environment

Install the real-data extras before any baseline run:

```bash
uv sync --extra dev --extra data
```

W&B tracking is enabled automatically when `WANDB_API_KEY` is present and tracking is not
explicitly disabled. The default entity is `achbogga-track`. You can override the entity with
`WANDB_ENTITY` if needed. To disable tracking for a local run, set `TSQBEV_WANDB=0` or
`WANDB_MODE=disabled`.
The shell that launches the experiment must actually see those environment variables; ad-hoc
exports in a different terminal session will not reach a separate non-interactive launcher.

Project naming keeps hyperparameter and performance tuning grouped within the same architecture
family. Materially different architecture families are logged to separate W&B projects so ablations
stay comparable without overloading one project namespace.

## Data Readiness Check

You can sanity-check a candidate root quickly:

```bash
uv run tsqbev check-data --dataset-root /path/to/root
```

This is a structural check only. It does not verify dataset licenses, corrupted archives, or completeness beyond the expected top-level files and folders.

## nuScenes Mini Baseline

Train the object-detection baseline on `v1.0-mini`:

```bash
uv run tsqbev train-nuscenes \
  --dataset-root /path/to/nuscenes \
  --artifact-dir artifacts/baselines \
  --preset rtx5000-nuscenes \
  --version v1.0-mini \
  --train-split mini_train \
  --split mini_val \
  --epochs 4 \
  --lr 3e-4 \
  --batch-size 2 \
  --grad-accum-steps 2
```

Optional cached teacher-guided training:

```bash
uv run tsqbev cache-teacher-nuscenes \
  --dataset-root /path/to/nuscenes \
  --version v1.0-mini \
  --result-json /path/to/external_teacher_results.json \
  --teacher-cache-dir /path/to/teacher-cache
```

This converts a standard nuScenes detection result JSON from an external detector into the local
`TeacherCacheStore` format expected by the repo.

Verified external teacher bootstrap:

- public teacher: OpenPCDet `CenterPoint-PointPillar`
- measured external `v1.0-mini` result: `mAP 0.4369`, `NDS 0.4997`
- measured repo-local cache audit on `mini_train`: `323 / 323` records, `coverage = 1.0`
- measured repo-local cache audit on `mini_val`: `81 / 81` records, `coverage = 1.0`
- benchmark details:
  [`docs/benchmarks/openpcdet-centerpoint-mini.md`](openpcdet-centerpoint-mini.md)

Then train from the cache:

```bash
uv run tsqbev train-nuscenes \
  --dataset-root /path/to/nuscenes \
  --artifact-dir artifacts/baselines \
  --version v1.0-mini \
  --train-split mini_train \
  --split mini_val \
  --epochs 4 \
  --lr 3e-4 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --teacher-kind cache \
  --teacher-cache-dir /path/to/teacher-cache
```

This expects cached repo-local `TeacherTargets`, not a live heavyweight LiDAR framework in the
default runtime. The teacher bootstrap contract is documented in
[`specs/006-lidar-teacher-bootstrap.md`](../specs/006-lidar-teacher-bootstrap.md).

Audit teacher-cache coverage before claiming a teacher-lift result:

```bash
uv run tsqbev audit-teacher-cache-nuscenes \
  --dataset-root /path/to/nuscenes \
  --version v1.0-mini \
  --split mini_train \
  --teacher-cache-dir /path/to/teacher-cache \
  --output-dir artifacts/teacher_cache_audit_train

uv run tsqbev audit-teacher-cache-nuscenes \
  --dataset-root /path/to/nuscenes \
  --version v1.0-mini \
  --split mini_val \
  --teacher-cache-dir /path/to/teacher-cache \
  --output-dir artifacts/teacher_cache_audit_val
```

The exact external OpenPCDet `CenterPoint-PointPillar` runbook is in
[`docs/openpcdet-centerpoint-teacher.md`](openpcdet-centerpoint-teacher.md), and the measured
teacher benchmark is in [`docs/benchmarks/openpcdet-centerpoint-mini.md`](openpcdet-centerpoint-mini.md).

The first paired teacher-on versus teacher-off bounded mini invocation is currently writing to:

- `artifacts/research_teacher_v1/research_loop_teacher_v1.log`
- `artifacts/research_teacher_v1/research_loop/`

Before trying that path on any machine, run the prerequisite check:

```bash
uv run tsqbev check-openpcdet-env \
  --openpcdet-repo-root /path/to/OpenPCDet
```

If the cached teacher outputs include `object_boxes`, `object_labels`, and `object_scores`, the
teacher-enabled preset replaces the raw LiDAR seed path with projected teacher seeds while still
keeping the heavy teacher itself outside the default runtime.

Run the bounded local research loop on `v1.0-mini`:

```bash
uv run tsqbev research-loop \
  --dataset-root /path/to/nuscenes \
  --artifact-dir artifacts/baselines \
  --max-experiments 5 \
  --device cuda
```

The loop now follows a staged `baseline -> explore -> exploit` pattern inspired by the public
`karpathy/autoresearch` workflow, but adapted to this repo's narrower ML contract. Each run emits a
JSONL ledger, a TSV ledger, a per-run manifest, and a machine-readable scale verdict. The current
bounded contract also uses a fixed comparable `max_train_steps=960` budget per recipe.
When W&B is available, the same bounded loop mirrors metrics and metadata to W&B under the project
derived for that architecture family. Tracking failures never abort training or evaluation.

Export predictions for official local validation:

```bash
uv run tsqbev export-nuscenes \
  --dataset-root /path/to/nuscenes \
  --version v1.0-mini \
  --split mini_val \
  --output-path artifacts/eval/nuscenes_predictions.json
```

Run the official local nuScenes validation metrics:

```bash
uv run tsqbev eval-nuscenes \
  --dataset-root /path/to/nuscenes \
  --version v1.0-mini \
  --split mini_val \
  --output-path artifacts/eval/nuscenes_predictions.json \
  --output-dir artifacts/eval
```

Expected metric family from the official devkit:

- `mAP`
- `NDS`
- `mATE`
- `mASE`
- `mAOE`
- `mAVE`
- `mAAE`

## nuScenes Overfit Gate

Run the dedicated 32-sample overfit gate on the current promoted mini architecture:

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
  --device cuda
```

This command trains and evaluates on the exact same fixed token subset and writes:

- `artifacts/gates/overfit_gate/subset_tokens.json`
- `artifacts/gates/overfit_gate/summary.json`
- `artifacts/gates/overfit_gate/eval/metrics_summary.json`

The evaluation path is still grounded in the official nuScenes metric stack, but restricted to the
explicit token subset instead of an entire named split.

First recorded result:

- verdict: fail
- train-total ratio: `0.5079`
- same-subset official `NDS`: `0.0003752`
- same-subset official `mAP`: `0.0007504`
- `car AP @ 4.0m`: `0.0`
- detailed note: [docs/benchmarks/nuscenes-overfit-gate.md](benchmarks/nuscenes-overfit-gate.md)

Latest measured overfit-gate result on the current promoted mini architecture:

| Metric | Value |
| --- | ---: |
| Final train total / initial train total | `0.5079` |
| Same-subset official `NDS` | `0.0003752` |
| Same-subset official `mAP` | `0.0007504` |
| Nonzero classes | `1` |
| `car AP @ 4.0m` | `0.0` |
| Decision | fail |

Artifact locations:

- `artifacts/gates/overfit_gate/summary.json`
- `artifacts/gates/overfit_gate/eval/metrics_summary.json`

## Recorded Mini Results

Historical first bounded research-loop sweep on `v1.0-mini`:

| Recipe | Val Total | Synthetic Mean ms | Decision |
| --- | ---: | ---: | --- |
| `mini_mbv3_frozen_bs2` | 28.0525 | 15.6993 | keep |
| `mini_mbv3_frozen_bs4` | 29.4311 | 15.6731 | discard |
| `mini_mbv3_unfrozen_bs2` | 29.6107 | 15.6858 | discard |

Promoted 4-epoch mini baseline:

| Metric | Value |
| --- | ---: |
| Final val total | 24.4006 |
| Official `mAP` | `1.5376e-05` |
| Official `NDS` | `7.6880e-06` |
| Nonzero class result | `car AP @ 4.0m = 6.1504e-04` |

Artifact locations:

- sweep ledger: `artifacts/baselines/research_loop/results.jsonl`
- sweep ledger TSV: `artifacts/baselines/research_loop/results.tsv`
- sweep summary: `artifacts/baselines/research_loop/summary.json`
- promoted baseline history: `artifacts/baselines/mini_selected/nuscenes/history.json`
- promoted eval summary: `artifacts/baselines/mini_selected/eval/nuscenes/metrics_summary.json`

Strengthened bounded research-loop sweep on `v1.0-mini`:

| Recipe | Stage | Val Total | Official mAP | Official NDS | Mean ms | Source Mix | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `mini_balanced_mbv3_frozen` | baseline | 22.0419 | 0.0 | 0.0 | 17.4492 | `50 / 33 / 17` | discard |
| `mini_propheavy_mbv3_frozen` | explore | 23.0423 | `3.1083e-04` | `1.5541e-04` | 17.1797 | `33 / 50 / 17` | discard |
| `mini_propheavy_effb0_frozen` | explore | 24.1645 | 0.0 | 0.0 | 21.6682 | `33 / 50 / 17` | discard |
| `mini_propheavy_mbv3_frozen_query_boost` | exploit | 20.1352 | `1.1140e-04` | `0.0158` | 17.1938 | `31 / 54 / 15` | promote |
| `mini_propheavy_mbv3_frozen_lr_down` | exploit | 24.7614 | 0.0 | 0.0 | 17.1298 | `33 / 50 / 17` | discard |

Key changes:

- the strengthened loop now selects by official `mini_val` `NDS`, then `mAP`, then validation
  loss only as a tiebreaker
- the promoted recipe came from bounded exploitation around the incumbent, not from the initial
  flat exploration pass

This prevented the repo from incorrectly promoting the lowest-loss recipe, which still had
`NDS = 0.0`, and it produced the current best promoted public mini run.

Current answer on scale:

- do **not** scale by 10x compute yet
- current best completed official `mini_val` result is `NDS = 0.0158068933`, `mAP = 1.1140e-04`
- no passing overfit-gate artifact is recorded yet
- the external teacher bootstrap is now verified, but no paired student teacher-lift artifact is
  recorded yet
- see [docs/scaling-gates.md](scaling-gates.md) for the required promotion thresholds

Artifact locations for the strengthened loop:

- sweep ledger: `artifacts/research_v3/research_loop/results.jsonl`
- sweep ledger TSV: `artifacts/research_v3/research_loop/results.tsv`
- sweep summary: `artifacts/research_v3/research_loop/summary.json`

## OpenLane Baseline

Train the lane baseline on the small official split first:

```bash
uv run tsqbev train-openlane \
  --dataset-root /path/to/openlane \
  --artifact-dir artifacts/baselines \
  --subset lane3d_300 \
  --epochs 6 \
  --lr 2e-4 \
  --grad-accum-steps 8
```

Export lane predictions:

```bash
uv run tsqbev export-openlane \
  --dataset-root /path/to/openlane \
  --subset lane3d_300 \
  --output-dir artifacts/eval
```

Run the official OpenLane 3D lane evaluation:

```bash
uv run tsqbev eval-openlane \
  --dataset-root /path/to/openlane \
  --subset lane3d_300 \
  --openlane-repo-root /tmp/OpenLane \
  --test-list /path/to/openlane_test_list.txt \
  --output-dir artifacts/eval
```

Expected metric family from the official kit:

- `F-score`
- `Recall`
- `Precision`
- `Category Accuracy`
- `x error close`
- `x error far`
- `z error close`
- `z error far`

## Current State

What is ready now:

- real public-data loaders
- set-based matching losses instead of the original smoke losses
- local training entrypoints
- official local evaluation wrappers for `nuScenes` and `OpenLane`
- a bounded `nuScenes v1.0-mini` research loop
- repo-local validation still green

What is still pending:

- any future full `v1.0-trainval` promotion, if desired
- stronger training schedules, if the goal moves from a functional baseline toward a competitive one
- pretrained external LiDAR teacher ablations through the optional teacher cache/provider path
