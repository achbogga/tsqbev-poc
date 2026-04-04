# Foundation-Teacher Perspective-Sparse Reset

This document captures the current architecture reset decision.

The repo's legacy sparse-query line remains useful as a control, but the primary target is no
longer "pure dense-BEV fusion as the runtime." The new target is a perspective-supervised sparse
temporal student with unrestricted foundation teachers and a staged lane/vector-map side head,
aimed at AGX Orin deployment.

## Why The Direction Changed

The repo now has three hard pieces of evidence:

- the official public BEVFusion detection baseline was reproduced locally at
  `mAP 0.6730 / NDS 0.7072`, so the dense-BEV control ceiling is real
- the local lightweight sparse student can improve, but its best `mini_val` frontier is still far
  below the reproduced public ceiling
- recent public camera-only temporal sparse methods are now stronger than "plain BEVFusion as the
  answer," especially when they use stronger image backbones and sparse temporal aggregation

The reset therefore needs to be more precise than "move to BEVFusion." The evidence-backed move is:

- keep BEVFusion and OpenPCDet as teacher and control stacks
- move the runtime student toward sparse temporal multi-view aggregation
- add perspective supervision so strong modern image backbones adapt to 3D reasoning
- inject foundation features before the student camera branch instead of training the camera side
  from scratch

## Primary Evidence

- [Sparse4Dv3](https://github.com/HorizonRobotics/Sparse4D) reports `0.656 NDS / 0.570 mAP` on
  nuScenes test with a public camera-only sparse temporal pipeline, and
  `0.719 NDS / 0.668 mAP` for the offline EVA02-Large variant with external data. That is direct
  evidence that sparse temporal perception with stronger image priors is a stronger frontier than
  treating BEVFusion alone as the destination.
- [BEVFormer v2](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_Recognition_via_CVPR_2023_paper.pdf)
  argues that pure BEV supervision gives weak guidance to modern image backbones and adds
  perspective supervision explicitly to adapt them to 3D reasoning without cumbersome depth
  pretraining.
- [DINOv2](https://github.com/facebookresearch/dinov2) provides public hub-loadable pretrained
  backbones, and [DINOv3](https://github.com/facebookresearch/dinov3) now provides dense-feature
  models plus distilled ViT and ConvNeXt variants with public code and weight-access instructions.
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) still provides the most practical public
  LiDAR and multimodal teacher suite, including CenterPoint, TransFusion, and BEVFusion checkpoints.
- [MapTRv2](https://github.com/hustvl/MapTR) remains the best public lane/vector-map head to add
  once the shared perception latent is stable.
- NVIDIA's public Alpamayo announcements position Alpamayo as an open-reasoning autonomous-driving
  model used for long-tail autonomous driving development and evaluation, which fits this repo as a
  teacher/evaluator layer, not as the in-vehicle perception runtime.

## Final Target Stack

### Runtime Student

- camera trunk: frozen or lightly tuned `DINOv2` or `DINOv3` features projected into the camera
  branch with a small linear projector
- perception core: `Sparse4D`-style sparse temporal multi-view aggregation with propagated
  instances and sparse attention
- perspective auxiliary path: `BEVFormer v2`-style perspective supervision head to adapt the image
  backbone and stabilize 3D geometry
- LiDAR grounding: lightweight PointPillar / CenterPoint-style LiDAR anchor prior, not a heavy
  fused dense trunk as the student default
- lane/vector head: `MapTRv2`-style head added on the shared latent after the isolated lane branch
  is healthy
- deployment specialization: `EfficientViT`, then `OFA / AMC / HAQ`, plus activation checkpointing
  and auto-fit behavior to stay within local GPU RAM

### Unrestricted Teacher Suite

- geometry teacher: `OpenPCDet` `CenterPoint` / `BEVFusion`
- camera teacher: `Sparse4Dv3-offline` or equivalent strong sparse camera-only model with modern
  image priors
- dense feature teacher: `DINOv2` immediately, `DINOv3` as the stronger dense-feature path
- reasoning teacher: `NVIDIA Alpamayo` for hard-case mining, scenario critique, and teacher-side
  auto-evaluation, not as the student runtime trunk

## Distillation Priorities

The new KD priority order is:

1. teacher quality-aware ranking outputs and score calibration
2. projected camera features from DINOv2/DINOv3 into the student camera branch
3. perspective-head outputs and proposal quality targets
4. BEV-space teacher maps and dense teacher outputs from strong LiDAR/multimodal teachers
5. lane/map token distillation after the lane branch is real

This keeps the student small while letting the teacher be as large and expensive as needed.

## What BEVFusion Still Is

`BEVFusion` remains critical in this repo, but its role changed:

- it is a reproduced public control baseline
- it is a strong teacher and deployment reference
- it is not the final student architecture thesis

## What The Repo Should Stop Doing

- stop treating pure dense-BEV fusion as the only migration target
- stop reopening tiny schedule and query-budget tweaks after two stalled winner-line runs
- stop assuming lane should be mixed in before the shared perception latent is stable

## What The Repo Should Do Next

1. keep the current sparse teacher-quality line only until it clearly stalls
2. add the foundation camera projector path with `DINOv2` first and `DINOv3` second
3. add a perspective-supervision auxiliary head to the student camera branch
4. keep `BEVFusion` / `OpenPCDet` as dense BEV teachers and controls
5. bring up `OpenLane` in isolation, then add `MapTRv2`-style vector supervision with a
   detection non-regression gate
