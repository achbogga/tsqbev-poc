# Dense-BEV Reset Stack

This document captures the architecture reset decision.

The repo’s legacy sparse-query line remains useful as a comparison control, but the primary target
is now a dense BEV fusion stack assembled from public upstreams and aimed at AGX Orin deployment.

## Why Move

The current repo already proved three things:

- the public data and evaluation plumbing works
- the external teacher path is real and reproducible
- the custom sparse-query student is still not the best-available path for a strong deployable
  baseline

The gap is not more local tuning around query seeds. The gap is that the repo is still missing the
mature shared-BEV substrate that the strongest public multimodal systems already use.

## Target Stack

- LiDAR branch: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and
  [CenterPoint](https://github.com/tianweiy/CenterPoint)
- camera BEV branch: [BEVDet / BEVDepth](https://github.com/HuangJunJie2017/BEVDet)
- shared BEV fusion: [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- dense detection head: CenterPoint / CenterHead-style dense BEV detection
- lane and map head: [MapTR / MapTRv2](https://github.com/hustvl/MapTR)
- lane transfer reference: [PersFormer](https://github.com/OpenDriveLab/PersFormer_3DLane)
- optional dense priors: [DINOv2](https://github.com/facebookresearch/dinov2) and
  [DINOv3](https://github.com/facebookresearch/dinov3)
- deployment specialization: [EfficientViT](https://github.com/mit-han-lab/efficientvit), then
  [OFA](https://hanlab.mit.edu/projects/ofa), [AMC](https://hanlab.mit.edu/projects/amc), and
  [HAQ](https://hanlab.mit.edu/projects/haq)
- Orin deployment substrate:
  [NVIDIA DS3D BEVFusion docs](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_3D_MultiModal_Lidar_Camera_BEVFusion.html)

## Gap Analysis

Current repo strengths:

- strong contracts and tests
- good teacher-cache and evaluation scaffolding
- measured latency harnesses
- bounded research-loop discipline

Current repo gaps versus the reset target:

- no shared dense BEV trunk as the primary representation
- no public checkpoint-backed BEVFusion / BEVDet / BEVDepth reproduction path yet
- no mature dense lane / map head on the same BEV representation
- no official Orin deployment story for the new target stack yet
- too much custom sparse-query machinery for the current evidence level

## Why This Is the Better Default

The reset stack is the higher-ROI direction because it:

- starts from public checkpoints and working code instead of inventing the full geometry stack
- keeps detection and lane/map equally supported by one shared BEV representation
- fits a clean deployment path through TensorRT and DeepStream on Orin
- leaves the custom sparse-query line available only as a historical comparison baseline

The legacy line can still be useful for ablations and for comparing object-centric versus dense-BEV
representations, but it is no longer the architecture that should define the repo.
