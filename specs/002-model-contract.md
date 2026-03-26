# Spec 002: Model Contract

## Goal

Implement a minimal multimodal temporal sparse-query system with evidence-backed design choices.

## Fixed v1 Decisions

- lightweight pillar LiDAR encoder
- tri-source query initialization: LiDAR, 2D proposal-ray, global learned seeds
- sparse camera feature sampling
- streaming temporal state
- object detection head
- camera-dominant lane head with LiDAR ground hints
- optional map-token fusion

## Non-Goals For v1

- dense multimodal BEV fusion as the main path
- distributed training
- always-on autonomous mutation

## References

- DETR3D sparse sampling: <https://proceedings.mlr.press/v164/wang22b/wang22b.pdf>
- PETR / PETRv2 query and temporal framing: <https://github.com/megvii-research/PETR>
- Sparse4D sparse fusion efficiency: <https://arxiv.org/pdf/2211.10581>
- PersFormer lane reasoning: <https://arxiv.org/abs/2203.11089>
