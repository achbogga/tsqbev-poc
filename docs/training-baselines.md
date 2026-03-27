# Public Baseline Workflow

This repository now contains real-data training and evaluation plumbing for the public baselines, but it does **not** yet publish full-dataset accuracy numbers because the required public datasets are not present on the current machine.

Grounding for the acquisition requirements:

- `nuScenes` download and devkit structure: <https://github.com/nutonomy/nuscenes-devkit>
- `OpenLane V1` dataset structure and evaluation kit: <https://github.com/OpenDriveLab/OpenLane>

## Required Public Data

### nuScenes

The code expects a standard `nuScenes` dataroot containing at least:

```text
<nuscenes-root>/
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

The official devkit states that for full `nuScenes` use you should download the dataset from the official download page and that the devkit expects all archives for the full setup.

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

## nuScenes Baseline

Train the object-detection baseline:

```bash
uv run tsqbev train-nuscenes \
  --dataset-root /path/to/nuscenes \
  --artifact-dir artifacts/baselines \
  --version v1.0-trainval \
  --split val \
  --epochs 4 \
  --lr 3e-4 \
  --grad-accum-steps 8
```

Export predictions for official local validation:

```bash
uv run tsqbev export-nuscenes \
  --dataset-root /path/to/nuscenes \
  --version v1.0-trainval \
  --split val \
  --output-path artifacts/eval/nuscenes_predictions.json
```

Run the official local nuScenes validation metrics:

```bash
uv run tsqbev eval-nuscenes \
  --dataset-root /path/to/nuscenes \
  --version v1.0-trainval \
  --split val \
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
- repo-local validation still green

What is still pending:

- actual full-dataset runs on this machine
- measured baseline accuracy tables in the paper
- tuned baseline numbers in the docs

Those final numbers should only be published after the real datasets are present and the runs complete successfully.
