# tsqbev-poc

`tsqbev-poc` is a public proof-of-concept repository for a multimodal temporal sparse-query BEV system:

- LiDAR grounds 3D object anchors and depth
- cameras provide appearance, semantics, and lane structure
- map priors are optional
- temporal state is sparse and streaming
- distillation is designed in from the start

This repo is intentionally small and evidence-driven. Every module is tied back to an original paper and, where available, an official codebase. See [docs/reference-matrix.md](docs/reference-matrix.md).

The local summary PDF in `/home/achbogga/projects/Production-ready Temporal Sparse Query BEV for Torc on NVIDIA Orin.pdf` is treated as internal synthesis only. The repo cites the underlying original papers, official codebases, and our own local paper/repo artifacts instead of citing that generated PDF directly.

## Status

- bootstrap in progress
- auto-research loop disabled
- local-first, CPU/synthetic-first validation
- GPU and Orin validation are separate acceptance stages

## Source Grounding

Primary references include:

- [DETR3D](https://proceedings.mlr.press/v164/wang22b.html)
- [PETR / PETRv2](https://github.com/megvii-research/PETR)
- [StreamPETR](https://github.com/exiawsh/StreamPETR)
- [Sparse4D](https://github.com/HorizonRobotics/Sparse4D)
- [SparseBEV](https://github.com/MCG-NJU/SparseBEV)
- [BEVDistill](https://arxiv.org/abs/2211.09386)
- [CMT](https://github.com/junjie18/CMT)
- [BEVFusion](https://arxiv.org/abs/2205.13542)
- [MapTR](https://github.com/hustvl/MapTR)
- [HotBEV](https://proceedings.neurips.cc/paper_files/paper/2023/file/081b08068e4733ae3e7ad019fe8d172f-Paper-Conference.pdf)

## Repo Layout

```text
docs/           plan and evidence trail
specs/          implementation contracts
src/tsqbev/     minimal multimodal implementation
tests/          isolated and integration tests
research/       intentionally disabled research loop scaffolding
artifacts/      local run outputs and exports
```

## Implementation Order

1. lock contracts and citations
2. implement geometry, LiDAR, and query seeding
3. implement the minimal multimodal model
4. add public/Torc-thin adapters
5. add export and latency harnesses
6. keep auto-research disabled until the repo is green and explicitly authorized

## Quick Start

After bootstrap is complete:

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
uv run pytest
uv run tsqbev smoke
uv run tsqbev train-step
uv run tsqbev eval
uv run tsqbev bench
```

For CUDA deployment validation on supported NVIDIA systems:

```bash
uv sync --extra dev --extra deploy
uv run tsqbev trt-bench
```

## Non-Goal For Now

This repo does not yet target:

- autonomous experimentation
- large-scale Ray orchestration
- final Orin deployment packaging
- full internal Torc data integration
