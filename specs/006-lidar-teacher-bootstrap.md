# Spec 006: External LiDAR Teacher Bootstrap

## Goal

Add optional support for a pretrained external LiDAR teacher so the repo can evaluate the novel
multimodal sparse-query design without first inventing a strong LiDAR backbone from scratch.

## Design Principle

The public student remains lightweight and dependency-light.

Heavy LiDAR stacks such as OpenPCDet, CenterPoint, or sparse-conv backbones must remain optional
and external to the default runtime. The default install path for `tsqbev-poc` must not require
heavyweight LiDAR framework dependencies.

## First Target Backend

The first teacher backend should target a public pretrained `CenterPoint-PointPillar` style model
through an optional external adapter.

Rationale:

- public nuScenes checkpoints exist
- pillar-based geometry is closer to the current student than a sparse-voxel stack
- it provides a credible LiDAR geometry prior without rewriting the core repo around sparse convs

## Teacher Integration Modes

### Mode A: Offline Cache

The teacher runs outside the core repo and writes cached tensors.

Required cacheable outputs:

- top-K object boxes in ego frame
- detection scores
- class ids or class logits
- optional per-query features
- optional router/logit priors

This is the preferred first mode because it keeps the student runtime simple and reproducible.

The first public ingress path must accept the standard nuScenes detection submission JSON written
by external/public teacher stacks and convert it into repo-local cache records.

### Mode B: Optional Online Adapter

The teacher may also be accessed through an optional import-guarded adapter, but only if:

- the adapter can fail gracefully when the external framework is absent
- the core repo still imports and tests cleanly without that framework

## Required Contracts

The repo must provide typed surfaces for:

- teacher backend config
- teacher detections
- teacher cache records
- serialization and deserialization
- conversion into `TeacherTargets` and optional sparse seed priors

Current implementation scaffold:

- `src/tsqbev/teacher_cache.py`
- `src/tsqbev/teacher_backends.py`
- `src/tsqbev/teacher_dataset.py`
- `src/tsqbev/teacher_seed.py`

## Required Experiments

At minimum, the repo must support the following teacher ablations:

1. teacher boxes as `Q_lidar` reference priors only
2. teacher boxes plus teacher scores as seed priors
3. teacher feature distillation into the student query bank

## Promotion Rule

The external teacher path is worth keeping only if it improves the student on official
`nuScenes v1.0-mini` validation by at least one of:

- teacher-cache audit coverage `>= 95%` on both `mini_train` and `mini_val`
- absolute `NDS` lift `>= +0.02`
- relative `NDS` lift `>= 2x`

The lift must be measured against the current student-only baseline on the same public mini setup.

## Forbidden

- making heavy external LiDAR frameworks mandatory in the default repo install
- silently changing the student architecture to mirror the teacher wholesale
- treating teacher-only metrics as student success
