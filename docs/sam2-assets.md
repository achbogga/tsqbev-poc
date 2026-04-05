# SAM 2.1 Assets

Local checkpoint cache for the TSQBEV frontier branch.

## Verified Files

- `/home/achbogga/projects/research/sam2_weights/sam2.1_hiera_base_plus.pt`
- `/home/achbogga/projects/research/sam2_weights/sam2.1_hiera_large.pt`

## Source

Downloaded from the official SAM 2.1 checkpoint endpoints published by the
`facebookresearch/sam2` repository:

- `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt`
- `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`

## Intended Use

- `sam2.1_hiera_base_plus.pt` is the first integration target for lightweight
  region / mask priors and promptable segmentation experiments.
- `sam2.1_hiera_large.pt` is the stronger teacher / upper-bound checkpoint for
  offline analysis, distillation experiments, and comparison runs where GPU
  memory allows it.

## Notes

- The shared cache directory is `/home/achbogga/projects/research/sam2_weights`.
- The repo should treat these as read-only assets and reference them by absolute
  path from config or run manifests.
