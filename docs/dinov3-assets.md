# DINOv3 Assets

Local checkpoint cache for the TSQBEV frontier camera branch.

## Verified Files

- `/home/achbogga/projects/research/dinov3_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- `/home/achbogga/projects/research/dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`

## Source

Downloaded from the official Meta DINOv3 gated download endpoints after access was
approved under the DINOv3 license.

## Intended Use

- `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` is the first local RTX 5000 pilot
  backbone for projected camera features with activation checkpointing enabled.
- `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` is the stronger follow-up branch
  for larger-teacher or higher-VRAM experiments once the ViT-S local path is
  validated end to end.

## Notes

- The shared cache directory is `/home/achbogga/projects/research/dinov3_weights`.
- The repo should prefer explicit absolute `foundation_weights` paths over remote
  torch-hub downloads for reproducibility and to avoid gated-download failures.
