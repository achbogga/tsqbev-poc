# Spec 008: Dense-BEV Reset Contract

## Goal

Define the migration target for `tsqbev-poc`: a dense BEV fusion stack assembled from public
upstreams, with detection and lane/map treated as equal-priority heads on a shared BEV
representation.

## Fixed Target Stack

- LiDAR encoder: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) /
  [CenterPoint](https://github.com/tianweiy/CenterPoint) / PointPillars
- camera BEV encoder: [BEVDet / BEVDepth](https://github.com/HuangJunJie2017/BEVDet)
- fusion trunk: [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- detection head: dense CenterPoint / CenterHead-style BEV detection
- lane / map head: [MapTR / MapTRv2](https://github.com/hustvl/MapTR)
- lane transfer reference: [PersFormer](https://github.com/OpenDriveLab/PersFormer_3DLane)
- optional dense feature priors: [DINOv2](https://github.com/facebookresearch/dinov2) and
  [DINOv3](https://github.com/facebookresearch/dinov3)
- deployment specialization: [EfficientViT](https://github.com/mit-han-lab/efficientvit),
  then [OFA](https://hanlab.mit.edu/projects/ofa), [AMC](https://hanlab.mit.edu/projects/amc),
  and [HAQ](https://hanlab.mit.edu/projects/haq)
- deployment target: TensorRT / DeepStream on NVIDIA AGX Orin

## Migration Rules

- The legacy sparse-query line remains in the repo as comparison evidence only.
- Dense-BEV baselines must be reproduced from public upstreams before any large-scale redesign is
  claimed successful.
- Upstream runtimes with incompatible dependency stacks must be isolated behind their official
  environment boundary. For BEVFusion, that means the official Docker path rather than folding the
  old OpenMMLab stack into the main repo environment.
- Public upstream baselines must carry their full dataset-side prerequisites. For BEVFusion on
  nuScenes, that includes the official map-expansion bundle under `maps/{basemap,expansion,prediction}`.
- Detection and lane/map are equal-priority tasks on the same shared BEV trunk.
- Public checkpoints and official code paths take priority over custom reimplementation when both
  are available.
- DINO-style dense priors are optional and should not displace the dense-BEV runtime baseline.

## Non-Goals

- replacing the repo with an opaque one-off integration
- adding a second incompatible batch contract
- making DINO or SAM the first runtime backbone
- dropping the existing teacher, data, or evaluation scaffolding

## References

- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [BEVDet / BEVDepth](https://github.com/HuangJunJie2017/BEVDet)
- [MapTR](https://github.com/hustvl/MapTR)
- [PersFormer](https://github.com/OpenDriveLab/PersFormer_3DLane)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [DINOv3](https://github.com/facebookresearch/dinov3)
- [EfficientViT](https://github.com/mit-han-lab/efficientvit)
- [OFA](https://hanlab.mit.edu/projects/ofa)
- [AMC](https://hanlab.mit.edu/projects/amc)
- [HAQ](https://hanlab.mit.edu/projects/haq)
- [NVIDIA DS3D BEVFusion docs](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_3D_MultiModal_Lidar_Camera_BEVFusion.html)
