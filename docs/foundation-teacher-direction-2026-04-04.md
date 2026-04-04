# Foundation-Teacher Direction: 2026-04-04

This note locks the next architecture direction from primary sources and current repo evidence.

## Decision

The repo should stop treating pure dense-BEV fusion as the default runtime target.

The new default direction is:

- a `Sparse4D`-style sparse temporal student
- a `BEVFormer v2`-style perspective auxiliary head
- frozen or lightly tuned `DINOv2` / `DINOv3` camera foundation features projected into the
  student camera branch
- strong `OpenPCDet` / `BEVFusion` geometry teachers
- a staged `MapTRv2` lane/vector-map head
- `Alpamayo` as teacher/evaluator only

## Why This Is The Right Call

### 1. Pure BEVFusion is a strong control, not the frontier student

The repo already reproduced the official public BEVFusion detection path at
`mAP 0.6730 / NDS 0.7072`, so BEVFusion remains a strong ceiling and teacher/control. But the
public literature now offers stronger or more flexible camera-centric temporal sparse lines.

### 2. Sparse temporal camera perception is a real frontier

`Sparse4D` public results matter here:

- public repo with configs and checkpoints
- `Sparse4Dv3` test result: `0.656 NDS / 0.570 mAP`
- `Sparse4Dv3-offline` with `EVA02-large`: `0.719 NDS / 0.668 mAP`

This is direct evidence that sparse temporal multi-view perception with stronger image priors is a
credible frontier and not an academic detour.

Source: <https://github.com/HorizonRobotics/Sparse4D>

### 3. Perspective supervision is the right bridge for modern image backbones

`BEVFormer v2` explicitly diagnoses that pure BEV supervision provides weak guidance to modern
image backbones and introduces a perspective 3D head to provide direct supervision to image
features. That matches the current repo need: our lightweight student needs stronger camera-side 3D
signal before BEV or sparse aggregation.

Source:
<https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_BEVFormer_v2_Adapting_Modern_Image_Backbones_to_Birds-Eye-View_Recognition_via_CVPR_2023_paper.pdf>

### 4. Foundation camera features are now practical

`DINOv2` offers public hub-loadable pretrained backbones and distilled small models. `DINOv3`
extends this with high-quality dense features and public distilled ViT / ConvNeXt variants,
including `ConvNeXt Tiny`.

That makes a frozen-teacher or projected-feature path practical right now, without waiting for
private checkpoints.

Sources:

- <https://github.com/facebookresearch/dinov2>
- <https://github.com/facebookresearch/dinov3>

### 5. The teacher should stay unrestricted

The student must remain deployable, but the teacher does not need that restriction.

The practical teacher suite is:

- `OpenPCDet` / `BEVFusion` for geometry and multimodal dense supervision
- `Sparse4Dv3-offline` for sparse camera-temporal supervision
- `DINOv2` / `DINOv3` for foundation camera features
- `Alpamayo` for scenario reasoning, hard-case mining, and teacher-side evaluation

`Alpamayo` should not define the in-vehicle perception trunk. Its public role is closer to a
reasoning system for long-tail autonomous driving development.

## What This Means For Lane Lines

Lane work should now be staged as:

1. isolated `OpenLane V1` baseline and evaluator artifact
2. `MapTRv2`-style vector head on the shared latent
3. joint detection+lane only with detection non-regression gates

This is more disciplined than bolting lane work onto the current detection loop too early.

## Immediate Action

- finish the current winner-line control run
- if it does not materially beat `v28`, stop weight-only nudges on that family
- open the `DINOv2` projector branch next
- open the perspective auxiliary branch immediately after
- keep the dense-BEV stacks as teacher/control arms, not as the only runtime plan
