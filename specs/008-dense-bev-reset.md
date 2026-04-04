# Spec 008: Foundation-Teacher Perspective-Sparse Reset Contract

## Goal

Define the migration target for `tsqbev-poc`: a perspective-supervised sparse temporal student
augmented by foundation camera features and unrestricted public teachers, with lane/map staged on
top of the same latent once the detection branch is healthy.

## Fixed Target Stack

- runtime student core: [Sparse4D](https://github.com/HorizonRobotics/Sparse4D)-style sparse
  temporal multi-view perception
- perspective supervision:
  [BEVFormer v2](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_Recognition_via_CVPR_2023_paper.pdf)
- camera foundation priors: [DINOv2](https://github.com/facebookresearch/dinov2) first,
  [DINOv3](https://github.com/facebookresearch/dinov3) second
- LiDAR anchor prior: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) /
  [CenterPoint](https://github.com/tianweiy/CenterPoint) / PointPillars
- geometry and multimodal teachers: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and
  [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- lane / map head: [MapTR / MapTRv2](https://github.com/hustvl/MapTR)
- lane transfer reference: [PersFormer](https://github.com/OpenDriveLab/PersFormer_3DLane)
- reasoning teacher/evaluator: NVIDIA Alpamayo
- deployment specialization: [EfficientViT](https://github.com/mit-han-lab/efficientvit),
  then [OFA](https://hanlab.mit.edu/projects/ofa), [AMC](https://hanlab.mit.edu/projects/amc),
  and [HAQ](https://hanlab.mit.edu/projects/haq)
- deployment target: TensorRT / DeepStream on NVIDIA AGX Orin for the student only

## Migration Rules

- The legacy sparse-query line remains in the repo as comparison evidence only.
- Dense-BEV baselines must be reproduced from public upstreams, but they now serve primarily as
  controls and teacher ceilings rather than as the default runtime thesis.
- Upstream runtimes with incompatible dependency stacks must be isolated behind their official
  environment boundary. For BEVFusion, that means the official Docker path rather than folding the
  old OpenMMLab stack into the main repo environment.
- Public upstream baselines must carry their full dataset-side prerequisites.
- Detection remains the first bottleneck to clear; lane/map stays staged until its isolated
  baseline is real.
- Public checkpoints and official code paths take priority over custom reimplementation when both
  are available.
- DINO-style dense priors are part of the default student design, not an optional afterthought.
- Alpamayo is permitted only as teacher/evaluator logic and hard-case mining, not as the runtime
  perception trunk.

## Non-Goals

- replacing the repo with an opaque one-off integration
- adding a second incompatible batch contract
- making BEVFusion the only target student architecture
- dropping the existing teacher, data, or evaluation scaffolding
- letting lane work silently regress the detection frontier

## References

- [Sparse4D](https://github.com/HorizonRobotics/Sparse4D)
- [BEVFormer v2](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_Recognition_via_CVPR_2023_paper.pdf)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [MapTR](https://github.com/hustvl/MapTR)
- [PersFormer](https://github.com/OpenDriveLab/PersFormer_3DLane)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [DINOv3](https://github.com/facebookresearch/dinov3)
- [EfficientViT](https://github.com/mit-han-lab/efficientvit)
- [OFA](https://hanlab.mit.edu/projects/ofa)
- [AMC](https://hanlab.mit.edu/projects/amc)
- [HAQ](https://hanlab.mit.edu/projects/haq)
- [NVIDIA DS3D BEVFusion docs](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_3D_MultiModal_Lidar_Camera_BEVFusion.html)
