# Spec 002: Legacy Sparse-Query Model Contract

## Goal

Document the legacy sparse-query system that remains in the repo as comparison evidence while the
reset stack is being migrated to a dense BEV fusion architecture. The new target stack is defined
in [Spec 008](./008-dense-bev-reset.md).

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
