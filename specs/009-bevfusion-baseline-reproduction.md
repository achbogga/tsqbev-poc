# Spec 009: BEVFusion Baseline Reproduction

## Goal

Reproduce the official public `BEVFusion` nuScenes baseline locally before any repo-native
dense-BEV student is treated as the new dense-BEV incumbent.

Primary sources:

- <https://github.com/mit-han-lab/bevfusion>
- <https://github.com/mit-han-lab/bevfusion/blob/main/docker/Dockerfile>
- <https://github.com/mit-han-lab/bevfusion/blob/main/configs/nuscenes/default.yaml>
- <https://github.com/mit-han-lab/bevfusion/issues/569>
- <https://zenodo.org/records/15667707>
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
- The public map-expansion recovery path may use the official nuScenes mirror at
  `https://zenodo.org/records/15667707`, but the repo must keep citing the upstream BEVFusion and
  nuScenes sources that make the dependency necessary.
- The repo-local eval wrapper may build `feature_decorator_ext` as a compatibility step because
  the archived upstream `setup.py` omits that extension from `ext_modules` even though
  `mmdet3d.ops.__init__` imports it unconditionally.
- The repo-local eval wrapper may prepend compatibility shims for import-only dependencies that are
  not used by the chosen camera+lidar config, such as `flash_attn` and `numba`, as long as those
  shims fail loudly if a radar or JIT-dependent path is actually instantiated.
- Repo-local compatibility shims may also supply minimal interpreter startup fixes for archived
  dependency combinations, such as restoring `numpy.long` for older codepaths.
- The repo-local reproduction path may patch archived upstream runtime inconsistencies in the
  mounted checkout at container startup when those inconsistencies prevent the official pretrained
  eval from running at all. Current allowed examples are:
  - fixing the `DepthLSSTransform` input-stem channel count so it matches the depth tensor
    emitted by the archived pipeline on this workstation
  - writing a derived compatibility checkpoint that preserves pretrained scalar-depth weights in
    channel `0` and zero-initializes the added feature channels
- This repo must not patch the archived BEVFusion source tree just to make eval work.
- The current local reproduction contract may force single-sample validation batches if the
  archived sparse `spconv` path is batch-unstable on this workstation and the single-sample path is
  otherwise healthy. This must be documented as a compatibility deviation, not hidden.
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
- the repo-local compatibility build for `feature_decorator_ext` succeeds when the upstream image
  does not already contain that artifact
- any local compatibility deviations from the archived upstream runtime are recorded explicitly in
  the reproduction report

The reset stack should not be declared reproduced until the local report records:

- upstream repo branch and `HEAD`
- exact config path
- exact checkpoint file used
- dataset root used
- official metrics emitted locally
- any deviation from the upstream published numbers
