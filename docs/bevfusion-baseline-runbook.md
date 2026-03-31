# BEVFusion Baseline Runbook

This document captures the official local reproduction path for the public BEVFusion nuScenes
baseline on this workstation.

Primary sources:

- [BEVFusion README](https://github.com/mit-han-lab/bevfusion)
- [BEVFusion Dockerfile](https://github.com/mit-han-lab/bevfusion/blob/main/docker/Dockerfile)
- [BEVFusion nuScenes config contract](https://github.com/mit-han-lab/bevfusion/blob/main/configs/nuscenes/default.yaml)
- [Archived official issue on the `create_data.py` failure mode](https://github.com/mit-han-lab/bevfusion/issues/569)
- [nuScenes devkit releases](https://github.com/nutonomy/nuscenes-devkit/releases)
- [nuScenes map expansion bundle mirror](https://zenodo.org/records/15667707)
- [NVIDIA DeepStream DS3D BEVFusion docs](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_3D_MultiModal_Lidar_Camera_BEVFusion.html)

## Why This Exists

The dense-BEV reset picked `BEVFusion` as the primary public multimodal runtime because it combines:

- an official public nuScenes detection checkpoint (`68.52 mAP`, `71.38 NDS`)
- an official public nuScenes map-segmentation checkpoint (`62.95 mIoU`)
- explicit NVIDIA TensorRT and Jetson Orin deployment references
- a shared-BEV interface that matches the repo reset target

The upstream repo is archived, so the reproduction path should be written down and checked, not
held in memory.

## Official Upstream Constraints

Per the upstream README and Dockerfile, the official stack is:

- Python `>=3.8, <3.9`
- PyTorch `1.10.1`
- CUDA `11.3`
- `mmcv==1.4.0`
- `mmcv-full==1.4.0`
- `mmdet==2.20.0`
- OpenMPI plus `mpi4py==3.0.3`

The official repo recommends Docker and points to `CUDA-BEVFusion` as the TensorRT best-practice
path, citing `25 FPS` on Jetson Orin.

## Current Local Reality

- Docker is installed and working.
- NVIDIA container support is installed and container GPU access works.
- nuScenes is extracted at `/mnt/storage/research/nuscenes`.
- The dataset root already contains `samples`, `sweeps`, `maps`, `maps/{basemap,expansion,prediction}`,
  and `v1.0-trainval`.
- The remaining prep step before official eval is creating the exact
  `nuscenes_infos_train.pkl` / `nuscenes_infos_val.pkl` files expected by BEVFusion configs.
- The official nuScenes map-expansion bundle was downloaded from the public Zenodo mirror linked
  above and extracted into `maps/{basemap,expansion,prediction}`. This was a hard prerequisite
  because BEVFusion's shared nuScenes pipeline instantiates `NuScenesMap` even for the detection
  config inherited from `configs/nuscenes/default.yaml`.
- The archived upstream `setup.py` omits `feature_decorator_ext` from its compiled extension list,
  while `mmdet3d.ops.__init__` imports it unconditionally. This repo's eval wrapper builds that
  missing compatibility extension before calling `tools/test.py`.
- The archived import surface also pulls in `flash_attn` and `numba` through unused codepaths for
  the selected non-radar config. The repo-local eval wrapper prepends compatibility shims for
  those imports so the published camera+lidar baseline can load without patching upstream source.
- The archived `DepthLSSTransform` path is internally inconsistent on this workstation: the runtime
  data path produces a six-channel depth tensor while the pretrained checkpoint and source expect a
  one-channel stem. The repo-local eval wrapper therefore:
  - patches `depth_lss.py` in the mounted upstream checkout at container startup so the stem input
    channels match the actual depth tensor contract
  - writes a patched checkpoint copy that preserves the pretrained scalar-depth stem weights in
    channel `0` and zero-initializes the added feature channels
- The archived sparse BEVFusion LiDAR path is also batch-sensitive on this workstation. Raw nuScenes
  points, raw voxelization, and the single-sample sparse encoder path all work, but the batched
  sparse path fails in `spconv` with `N > 0` assertions. The eval wrapper therefore forces
  `data.test.samples_per_gpu=1` and `data.samples_per_gpu=1` for the current reproduction path.

## Why The Repo Uses A Helper Instead Of `tools/create_data.py`

This repo does not treat upstream `tools/create_data.py` as the canonical eval-prep path.

Reason:

- the official configs expect `dataset_root + nuscenes_infos_train.pkl` and
  `dataset_root + nuscenes_infos_val.pkl`
- the archived official issue tracker documents a failure mode around the documented prep flow
- this repo therefore ships a narrow helper that uses the upstream converter and writes the exact
  ann-file names required for evaluation, without editing the upstream tree

## Machine-Readable Check

Use the repo-local checker to emit both readiness status and concrete commands:

```bash
cd /home/achbogga/projects/tsqbev-poc
uv run tsqbev check-bevfusion-env \
  --dataset-root /mnt/storage/research/nuscenes \
  --bevfusion-repo-root /home/achbogga/projects/bevfusion
```

## Repo-Local Helper

Use:

```bash
python /workspace/tsqbev-poc/research/scripts/prepare_bevfusion_nuscenes_infos.py \
  --bevfusion-root /workspace/bevfusion \
  --dataset-root /dataset \
  --version v1.0-trainval
```

This helper:

- imports the official upstream converter
- uses the official split logic
- writes `nuscenes_infos_train.pkl` and `nuscenes_infos_val.pkl` directly at the dataset root
- skips GT-database creation because pretrained evaluation does not need it

## Exact Commands

These wrappers are the preferred path because the raw upstream Dockerfile now fails on current
Conda Terms-of-Service enforcement unless the required `conda tos accept` commands are injected at
build time.

Build:

```bash
cd /home/achbogga/projects/tsqbev-poc
BEVFUSION_ROOT=/home/achbogga/projects/bevfusion \
  ./research/scripts/bootstrap_bevfusion_official.sh
```

Prepare ann files:

```bash
cd /home/achbogga/projects/tsqbev-poc
BEVFUSION_ROOT=/home/achbogga/projects/bevfusion \
DATASET_ROOT=/mnt/storage/research/nuscenes \
INFO_MODE=eval-only \
  ./research/scripts/run_bevfusion_nuscenes_prep.sh
```

Download the official map expansion if `maps/{basemap,expansion,prediction}` is still missing:

```bash
cd /home/achbogga/projects/tsqbev-poc
DATASET_ROOT=/mnt/storage/research/nuscenes \
  ./research/scripts/download_nuscenes_map_expansion.sh
```

Download checkpoints:

```bash
cd /home/achbogga/projects/tsqbev-poc
BEVFUSION_ROOT=/home/achbogga/projects/bevfusion \
  ./research/scripts/run_bevfusion_download_pretrained.sh
```

The eval wrapper will prepend the repo-local compat path and build the missing
`feature_decorator_ext` compatibility module automatically. If you need to do it explicitly:

```bash
docker run --rm --gpus all --shm-size 16g \
  -v /home/achbogga/projects/bevfusion:/workspace/bevfusion \
  -v /home/achbogga/projects/tsqbev-poc:/workspace/tsqbev-poc \
  -w /workspace/bevfusion \
  tsqbev-bevfusion-official:latest \
  /bin/bash -lc "export PYTHONPATH=/workspace/tsqbev-poc/compat:/workspace/bevfusion:\$PYTHONPATH && \
    python -m pip install ninja && \
    python /workspace/tsqbev-poc/research/scripts/build_bevfusion_feature_decorator_ext.py \
      --bevfusion-root /workspace/bevfusion"
```

Evaluate detection:

```bash
cd /home/achbogga/projects/tsqbev-poc
BEVFUSION_ROOT=/home/achbogga/projects/bevfusion \
DATASET_ROOT=/mnt/storage/research/nuscenes \
NUM_GPUS=1 \
  ./research/scripts/run_bevfusion_nuscenes_eval.sh
```

The current wrapper also:

- patches the archived `DepthLSSTransform` source in the mounted upstream checkout
- writes `pretrained/bevfusion-det.depthlss-compat.pth`
- forces single-sample val batches via `--cfg-options data.test.samples_per_gpu=1 data.samples_per_gpu=1`

Those are compatibility measures for the archived upstream on this workstation. They are not new
model ideas and should be treated as reproduction plumbing.

Evaluate segmentation:

```bash
cd /home/achbogga/projects/tsqbev-poc
BEVFUSION_ROOT=/home/achbogga/projects/bevfusion \
DATASET_ROOT=/mnt/storage/research/nuscenes \
NUM_GPUS=1 \
CONFIG_REL=configs/nuscenes/seg/fusion-bev256d2-lss.yaml \
CHECKPOINT_PATH=pretrained/bevfusion-seg.pth \
EVAL_KIND=map \
  ./research/scripts/run_bevfusion_nuscenes_eval.sh
```
