# Spec 001: Data Contracts

## Goal

Define the core multimodal tensors and metadata before model implementation.

## Required Types

- `SceneBatch`
- `QuerySeedBank`
- `MapPriorBatch`
- `TeacherTargets`
- `TemporalState`

## Required Behaviors

- shapes are validated
- LiDAR and camera data can coexist in the same batch
- 2D proposal seeds are optional but supported
- map priors are optional but supported
- teacher targets are optional but supported

## References

- PETRv2 multitask contracts: <https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_PETRv2_A_Unified_Framework_for_3D_Perception_from_MultiCamera_Images_ICCV_2023_paper.pdf>
- StreamPETR temporal state: <https://arxiv.org/abs/2303.11926>
- BEVDistill teacher-target interfaces: <https://arxiv.org/abs/2211.09386>
