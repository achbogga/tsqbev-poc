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
  --preset rtx5000-nuscenes-teacher \
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

If the cached teacher outputs include `object_boxes`, `object_labels`, and `object_scores`, the
teacher-enabled preset replaces the raw LiDAR seed path with projected teacher seeds while still
keeping the heavy teacher itself outside the default runtime.

Run the bounded local research loop on `v1.0-mini`:

```bash
uv run tsqbev research-loop \
  --dataset-root /path/to/nuscenes \
  --artifact-dir artifacts/baselines \
  --device cuda
```

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
- sweep summary: `artifacts/baselines/research_loop/summary.json`
- promoted baseline history: `artifacts/baselines/mini_selected/nuscenes/history.json`
- promoted eval summary: `artifacts/baselines/mini_selected/eval/nuscenes/metrics_summary.json`

Strengthened bounded research-loop sweep on `v1.0-mini`:

| Recipe | Val Total | Official mAP | Official NDS | Mean ms | Source Mix | Decision |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `mini_balanced_mbv3_frozen` | 20.9826 | 0.0 | 0.0 | 17.1130 | `50 / 33 / 17` | discard |
| `mini_propheavy_mbv3_frozen` | 22.4723 | 0.0 | 0.0 | 17.2225 | `33 / 50 / 17` | discard |
| `mini_propheavy_effb0_frozen` | 23.6836 | 0.0 | `0.0127` | 21.6604 | `33 / 50 / 17` | keep |

Key change:

- the strengthened loop now selects by official `mini_val` `NDS`, then `mAP`, then validation
  loss only as a tiebreaker

This prevented the repo from incorrectly promoting the lowest-loss recipe, which still had
`NDS = 0.0`.

Current answer on scale:

- do **not** scale by 10x compute yet
- current best official `mini_val` result is promising but still too weak
- see [docs/scaling-gates.md](scaling-gates.md) for the required promotion thresholds

Artifact locations for the strengthened loop:

- sweep ledger: `artifacts/research_v2/research_loop/results.jsonl`
- sweep summary: `artifacts/research_v2/research_loop/summary.json`

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
