# TSQBEV Frontier Program

This file is the compact, machine-readable summary of the frontier proposal. The research
supervisor should treat it as the active thesis unless exact local evidence falsifies part of it.

## Core Thesis

- The next meaningful frontier jump should come from a shared world latent, not another local
  schedule or query-budget tweak.
- The deployable student should stay Sparse4D-like and efficient.
- The strongest camera branch should use DINOv3 features with an explicit geometry bridge.
- The geometry bridge should use BEVFormer v2-style perspective supervision.
- Dense world-aligned supervision should come from unrestricted teachers such as BEVFusion and
  OpenPCDet.
- SAM 2.1 should be used as an offline region-support teacher, not as the runtime trunk.

## Immediate Stages

### S0 Control

- Hold the current trusted local control as the comparison baseline.
- Require official metrics and geometry sanity for every promotion.
- Do not reopen low-signal schedule-only branches.

### S1 Camera Bridge

- Continue only DINOv3 branches that include teacher distillation.
- Add perspective supervision to map image semantics into world geometry.
- Prefer quality-aware ranking and world-latent supervision over more local query mutations.

### S2 World Distillation

- Distill dense BEV or world features from BEVFusion or OpenPCDet teachers.
- Distill quality-aware class and box signals in world coordinates.
- Add SAM 2.1 region-support priors only as teacher-side supervision.

### S3 Lane Reintroduction

- Keep lane isolated or frozen-trunk only until detection is stably improved.
- Reintroduce lane through a staged MapTRv2-style vector head.
- Enforce detection non-regression before any joint promotion.

### S4 Efficiency

- After the new branch wins on official metrics, apply efficiency work:
  - activation checkpointing
  - automatic VRAM fit
  - EfficientViT, OFA, AMC, HAQ, AWQ style compression

## Explicit Non-Goals

- Do not resume naive joint detection plus lane training.
- Do not pivot to Mamba or other state-space models as the first fix for spatial/world alignment.
- Do not treat MoE or Alpamayo as the immediate deployable perception trunk.
- Do not claim progress from validation loss alone.

## Mandatory Gates

- Promotion requires official metric improvement over the trusted control.
- mAP may regress only within a small configured tolerance when NDS improves.
- Export sanity must remain physically plausible.
- If two consecutive proposal-driven runs fail to produce meaningful progress, pivot the architecture
  family or teacher family rather than repeating the same branch.

## Retrieval Hints

- DINOv3
- BEVFormer v2
- Sparse4D
- DistillBEV
- UniDistill
- BEVFusion
- OpenPCDet
- SAM 2.1
- MapTRv2
- quality-aware ranking
- world-aligned distillation
