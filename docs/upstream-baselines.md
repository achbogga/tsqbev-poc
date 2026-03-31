# Upstream Baselines

This document pins the public upstream baselines selected for the dense-BEV reset stack.

Primary sources:

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [BEVDet / BEVDepth](https://github.com/HuangJunJie2017/BEVDet)
- [MapTR / MapTRv2](https://github.com/hustvl/MapTR)
- [PersFormer](https://github.com/OpenDriveLab/PersFormer_3DLane)

## Bootstrap

Use the repo-local bootstrap script to clone the upstream repos into the shared projects directory
and pin `MapTR` to the `maptrv2` branch:

```bash
cd /home/achbogga/projects/tsqbev-poc
PROJECTS_ROOT=/home/achbogga/projects \
  ./research/scripts/bootstrap_reset_upstreams.sh
```

Then inspect the machine-readable manifest:

```bash
uv run tsqbev check-upstream-stack --projects-root /home/achbogga/projects
uv run tsqbev upstream-baselines --projects-root /home/achbogga/projects
uv run tsqbev check-bevfusion-env \
  --dataset-root /mnt/storage/research/nuscenes \
  --bevfusion-repo-root /home/achbogga/projects/bevfusion
```

The BEVFusion-specific bootstrap and evaluation path is documented in
[docs/bevfusion-baseline-runbook.md](bevfusion-baseline-runbook.md).

## Current Local Provenance

As of March 31, 2026, the local clones on this workstation are:

| Repo | Local path | Branch | HEAD |
| --- | --- | --- | --- |
| OpenPCDet | `/home/achbogga/projects/OpenPCDet_official` | `master` | `233f849829b6ac19afb8af8837a0246890908755` |
| BEVFusion | `/home/achbogga/projects/bevfusion` | `main` | `326653dc06e0938edf1aae7d01efcd158ba83de5` |
| BEVDet | `/home/achbogga/projects/BEVDet` | `dev3.0` | `26144be7c11c2972a8930d6ddd6471b8ea900d13` |
| MapTR | `/home/achbogga/projects/MapTR` | `maptrv2` | `e03f097abef19e1ba3fed5f471a8d80fbfa0a064` |
| PersFormer | `/home/achbogga/projects/PersFormer_3DLane` | `main` | `be280a4415e913baa5f884d9fb754b787d02c28f` |
| DINOv2 | `/home/achbogga/projects/dinov2` | `main` | `7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c` |
| DINOv3 | `/home/achbogga/projects/dinov3` | `main` | `31703e4cbf1ccb7c4a72daa1350405f86754b6d1` |
| EfficientViT | `/home/achbogga/projects/efficientvit` | `master` | `de7d7733cc0329f391b33f1f459271562ec27bd5` |

## Pinned Public Baselines

| Baseline | Repo | Config | Public checkpoint source | Reported metrics |
| --- | --- | --- | --- | --- |
| CenterPoint-PointPillar nuScenes | OpenPCDet | `tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml` | OpenPCDet model zoo | `mAP 50.03`, `NDS 60.70` |
| BEVFusion detection nuScenes | BEVFusion | `configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml` | BEVFusion README checkpoint link | `mAP 68.52`, `NDS 71.38` |
| BEVFusion segmentation nuScenes | BEVFusion | `configs/nuscenes/seg/fusion-bev256d2-lss.yaml` | BEVFusion README checkpoint link | `mIoU 62.95` |
| BEVDet R50 4D Depth CBGS | BEVDet | `configs/bevdet/bevdet-r50-4d-depth-cbgs.py` | BEVDet README checkpoint link | `mAP 36.1`, `NDS 48.3`, `FPS 25.2` |
| MapTRv2 nuScenes R50 24ep | MapTR | `projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py` | MapTR README checkpoint link | `mAP 61.4`, `FPS 14.1` |
| PersFormer OpenLane | PersFormer | `config/persformer_openlane.py` | no clearly linked checkpoint in current README | OpenLane F-score baseline documented in README |

## Current Reality

- The repos and pinned configs are now present locally.
- The OpenPCDet baseline is the only one already executed end to end in this repo.
- The BEVFusion official Docker path is now codified with a machine-readable checker and
  repo-local helper scripts.
- The official nuScenes map-expansion bundle is now present locally under
  `maps/{basemap,expansion,prediction}`.
- The archived BEVFusion runtime blockers are now handled by the repo-local wrapper via a missing
  `feature_decorator_ext` compatibility build plus guarded `flash_attn` / `numba` shims for the
  selected non-radar baseline.
- The active BEVFusion reproduction step is generating
  `nuscenes_infos_train.pkl` / `nuscenes_infos_val.pkl` with the repo-local eval-only helper.
- The next concrete work is environment-specific baseline reproduction for `BEVFusion`, `BEVDet`,
  `MapTRv2`, and `PersFormer`.
- For `BEVDet` and `MapTR`, the public checkpoint sources are still external download links rather
  than lightweight model-hub APIs.
