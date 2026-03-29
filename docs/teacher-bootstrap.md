# Teacher Bootstrap

This note explains how pretrained public LiDAR teachers fit into `tsqbev-poc`.

## Why A Teacher Path Exists

The current student is intentionally small. Its LiDAR path began as a lightweight pillar MLP so
that the public repo could stabilize quickly and keep deployment-oriented dependencies under
control.

That is the right engineering choice for the student. It is not the right way to estimate the
ceiling of the overall idea.

So the next step is not to bloat the student by default. The next step is to add an optional
pretrained external LiDAR teacher and measure how much our multimodal sparse-query design improves
once the geometry prior is stronger.

## Intended First Teacher

The first target is a public pretrained `CenterPoint-PointPillar` style teacher through an
optional external adapter or an offline cache export.

Why this is the first target:

- public nuScenes checkpoints exist
- pillar-based geometry is close to the current student path
- it is much easier to map teacher boxes and proposal priors into the current query bank than a
  full sparse-voxel teacher stack

Primary sources:

- CenterPoint paper: <https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.pdf>
- OpenPCDet model zoo: <https://github.com/open-mmlab/OpenPCDet>

## Ranked Candidate Order

1. `CenterPoint-PointPillar`
2. voxel `CenterPoint` as a stronger teacher-only ceiling check
3. `PillarNet` as a future stronger pillar-style encoder candidate
4. large-scale point-cloud pretraining methods only after the teacher path is proven

Why this order:

- `CenterPoint-PointPillar` has the best accuracy-to-integration-cost tradeoff for this repo
- voxel `CenterPoint` is stronger, but the sparse-conv dependency burden is higher
- `PillarNet` is appealing for a later student or teacher upgrade, but it is not the fastest
  route to an evidence-backed lift today
- generic point-cloud pretraining is a second-order optimization compared with first proving that
  a strong public LiDAR teacher materially helps our novel student design

Grounding:

- CenterPoint paper:
  <https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.pdf>
- CenterPoint repo:
  <https://github.com/tianweiy/CenterPoint>
- OpenPCDet model zoo:
  <https://github.com/open-mmlab/OpenPCDet>

## Ranked Public Options

### 1. CenterPoint-PointPillar

Best immediate teacher or seed source.

Why:

- strongest speed-to-evidence tradeoff for the current repo
- pillar-native representation is close to the student's LiDAR path
- public checkpoints are standard and widely reused

### 2. CenterPoint voxel variant

Best stronger teacher, not best first integration.

Why:

- likely stronger as a teacher
- substantially heavier integration burden
- better used externally than inside the student repo

### 3. PillarNet

Best future in-repo LiDAR encoder candidate if the teacher path validates the direction.

Grounding:

- PillarNet paper:
  <https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700034.pdf>
- PillarNet repo:
  <https://github.com/VISION-SJTU/PillarNet>

### 4. AD-PT / 3DTrans and similar LiDAR pretraining stacks

Interesting later, not the first bootstrap target.

Why:

- promising for longer-range teacher quality improvements
- slower path to immediate evidence on the current repo

## Preferred Execution Path

Start with offline caches, not a hard online dependency.

The teacher should export:

- top-K 3D boxes in ego frame
- scores
- class ids or class logits
- optional per-query features
- optional router priors

The student then consumes those through typed repo-local contracts.

Current repo-facing command pattern:

```bash
uv run tsqbev train-nuscenes \
  --dataset-root /path/to/nuscenes \
  --artifact-dir artifacts/teacher_bootstrap \
  --preset rtx5000-nuscenes-teacher \
  --version v1.0-mini \
  --train-split mini_train \
  --split mini_val \
  --teacher-kind cache \
  --teacher-cache-dir /path/to/teacher_cache \
  --epochs 6 \
  --batch-size 2 \
  --grad-accum-steps 2
```

This path keeps the heavy teacher framework outside the core repo. The teacher runs elsewhere,
writes cache records, and the student consumes those records through the optional cache/provider
surface.

The precise first external teacher runbook, grounded in the official OpenPCDet config and
evaluation/export path, is documented in
[docs/openpcdet-centerpoint-teacher.md](openpcdet-centerpoint-teacher.md).

Current repo surfaces:

- `src/tsqbev/teacher_cache.py`
- `src/tsqbev/teacher_backends.py`
- `src/tsqbev/teacher_dataset.py`
- `src/tsqbev/teacher_seed.py`

These provide:

- `.pt` teacher-cache serialization
- optional teacher-provider configs
- a cache-backed teacher provider
- a guarded external `OpenPCDet`/`CenterPoint` provider stub
- dataset wrapping to inject `TeacherTargets` into training batches
- teacher-seed projection from cached `object_boxes`, `object_labels`, and `object_scores`

## What Counts As Success

The teacher path is worth keeping only if it gives a material lift on official `nuScenes v1.0-mini`
validation.

Current minimum promotion rule:

- absolute `NDS` lift `>= +0.02`, or
- relative `NDS` lift `>= 2x`

Before any teacher-lift number counts, cache coverage must be audited and reach at least `95%` on
both `mini_train` and `mini_val`.

The formal contract is in [specs/006-lidar-teacher-bootstrap.md](../specs/006-lidar-teacher-bootstrap.md).
