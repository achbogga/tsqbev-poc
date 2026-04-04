# DINOv2 Camera Branch Decision

## Decision

The next high-ROI camera upgrade is a frozen `DINOv2` projector branch before BEV lifting, not another lightweight CNN tweak and not a `DINOv3` first implementation.

## Why This Branch

- `BEVFormer v2` argues that perspective supervision is the key missing signal when adapting strong 2D backbones to BEV recognition. We already moved the repo toward teacher-quality and perspective-aware ranking, so adding stronger perspective features is the next aligned step rather than another schedule-only mutation.
- `DINOv2` is the strongest immediately usable foundation camera path in the local environment because the official repo exposes stable local `torch.hub` loaders and intermediate spatial feature extraction.
- `DINOv3` remains promising, especially the ConvNeXt line, but it is not the first implementation path here because the local official repo currently requires extra deps beyond the repo venv, while `DINOv2` already loads and forwards.
- The student remains deployable because the `DINOv2` branch is frozen and projected down to the existing sparse-query model dimension. This is a research control arm, not the final runtime backbone claim.

## Evidence

- Official `DINOv2` local repo path worked with the current repo venv:
  - `/home/achbogga/projects/dinov2`
- Official `DINOv2` hub code exposes `dinov2_vits14_reg`, `dinov2_vitb14_reg`, and `get_intermediate_layers(..., reshape=True)` for dense spatial features:
  - `https://github.com/facebookresearch/dinov2`
- `BEVFormer v2` explicitly says perspective supervision is key for adapting modern image backbones to BEV:
  - `https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_Recognition_via_CVPR_2023_paper.pdf`
- `DINOv3` official repo is locally available but not yet a frictionless first path in this environment:
  - `https://github.com/facebookresearch/dinov3`

## Implementation in This Repo

- New backbone options:
  - `dinov2_vits14_reg`
  - `dinov2_vitb14_reg`
- New config controls:
  - `foundation_repo_root`
  - `foundation_intermediate_layers`
  - `foundation_patch_multiple`
  - `activation_checkpointing`
  - `attention_backend`
- New preset:
  - `rtx5000-nuscenes-dinov2-teacher`
- The lane branch now uses explicit SDPA-backed attention so backend choice is no longer implicit.

## Immediate Next Step

Run a bounded `nuScenes v1.0-mini` detection pilot with:

- frozen `DINOv2` camera projector
- current best teacher-quality supervision line
- official cached `CenterPoint-PointPillar` teacher targets
- constant-schedule mini budget

If that branch does not materially outperform the current `v28`/`v25` frontier, the next pivot should be denser teacher-output distillation or a stronger sparse temporal student, not more local weight nudges on MobileNet.
