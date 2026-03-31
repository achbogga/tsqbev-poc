# Spec 009: BEVFusion Baseline Reproduction

## Goal

Reproduce the official public `BEVFusion` nuScenes baseline locally before any repo-native
dense-BEV student is treated as the new dense-BEV incumbent.

Primary sources:

- <https://github.com/mit-han-lab/bevfusion>
- <https://github.com/mit-han-lab/bevfusion/blob/main/docker/Dockerfile>
- <https://github.com/mit-han-lab/bevfusion/blob/main/configs/nuscenes/default.yaml>
- <https://github.com/mit-han-lab/bevfusion/issues/569>
- <https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_3D_MultiModal_Lidar_Camera_BEVFusion.html>

## Contract

- Use the official upstream repo and official Docker path as the primary execution environment.
- Do not contaminate the core `uv` repo environment with the legacy `mmcv` / `mmdet` stack.
- Treat the upstream Dockerfile pins as the baseline source of truth:
  - Python `>=3.8, <3.9`
  - PyTorch `1.10.1`
  - CUDA `11.3`
  - `mmcv==1.4.0`
  - `mmcv-full==1.4.0`
  - `mmdet==2.20.0`
  - OpenMPI plus `mpi4py==3.0.3`
- The dataset root for BEVFusion must contain:
  - `samples/`
  - `sweeps/`
  - `maps/`
  - `maps/basemap/`
  - `maps/expansion/`
  - `maps/prediction/`
  - `v1.0-trainval/`
  - `nuscenes_infos_train.pkl`
  - `nuscenes_infos_val.pkl`
- Treat the nuScenes map-expansion bundle as a hard prerequisite for both detection and
  segmentation reproduction, because BEVFusion's shared nuScenes pipeline instantiates
  `NuScenesMap` even for the detection config inherited from `configs/nuscenes/default.yaml`.
- This repo must not patch the archived BEVFusion source tree just to make eval work.
- The repo-local ann-file helper may be used, but it must:
  - use the official upstream converter and split logic
  - emit the exact ann-file names expected by `configs/nuscenes/default.yaml`
  - avoid GT-database creation unless a training path explicitly needs it

## Acceptance

The BEVFusion reproduction path is considered operational only when all of these are true:

- container GPU access works
- `python setup.py develop` succeeds inside the official container
- repo-local ann-file generation succeeds on local nuScenes data
- official pretrained checkpoints download successfully
- the official nuScenes detection eval command runs to completion locally

The reset stack should not be declared reproduced until the local report records:

- upstream repo branch and `HEAD`
- exact config path
- exact checkpoint file used
- dataset root used
- official metrics emitted locally
- any deviation from the upstream published numbers
