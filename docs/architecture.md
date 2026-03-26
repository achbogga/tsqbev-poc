# Architecture

`tsqbev-poc` is a minimal multimodal BEV stack built around sparse object queries instead of a dense recurrent BEV tensor. The current implementation focuses on:

- LiDAR-grounded object initialization
- camera-driven sparse refinement
- persistent temporal state
- camera-dominant lane reasoning
- optional map priors
- exportability and deployment measurement

The underlying public references are indexed in [reference-matrix.md](./reference-matrix.md).

## System Overview

```mermaid
flowchart LR
    A[Multi-view images] --> B[Image backbone + 2-scale neck]
    C[LiDAR points] --> D[Lightweight pillar encoder]
    B --> E[2D proposal head]
    E --> F[Proposal-ray seed initializer]
    G[Learned global seeds] --> H[Tri-source query router]
    D --> H
    F --> H
    B --> I[Sparse cross-view sampler]
    H --> I
    I --> J[Query fusion block]
    K[Temporal state t-1] --> L[Temporal updater]
    J --> L
    L --> M[Object head]
    L --> N[Lane head]
    O[Optional map priors] --> N
    L --> P[Temporal state t]
```

## Query Lifecycle

The object pathway uses three seed sources:

- `Q_lidar`: geometric anchors from the LiDAR pillar encoder
- `Q_2d`: camera proposal seeds backprojected along calibrated rays
- `Q_global`: learned recovery anchors for recall

The router scores and filters the concatenated seed bank before sparse image sampling. This keeps the runtime bounded and follows the sparse-query design direction of DETR3D, PETR/PETRv2, and Sparse4D.

```mermaid
flowchart TD
    A[Q_lidar] --> D[Concatenate]
    B[Q_2d] --> D
    C[Q_global] --> D
    D --> E[Add source embeddings]
    E --> F[Router score head]
    F --> G[Top-k keep]
    G --> H[Sparse camera sampling]
    H --> I[Query fusion]
    I --> J[Temporal update]
    J --> K[Object boxes + classes]
```

## Deployment Split

The current public repo measures two paths:

- the full PyTorch model, including LiDAR seed extraction
- the exportable deployment core, which accepts prepared sparse seeds

This separation is intentional. It keeps the deployable graph compact and TensorRT-friendly while the public POC is still stabilizing.

```mermaid
flowchart LR
    subgraph FullModel[Full model path]
        A[LiDAR points] --> B[Pillar encoder]
        C[Images] --> D[Backbone + proposal path]
        B --> E[Core model]
        D --> E
        E --> F[Objects + lanes]
    end

    subgraph DeployCore[Exportable deployment core]
        G[Prepared LiDAR query seeds] --> H[Exportable core]
        I[Prepared proposal-ray seeds] --> H
        J[Image features] --> H
        H --> K[Objects + lanes]
    end
```

## Public Dataset Scope

The public repo currently targets:

- `nuScenes` for 3D object detection
- `OpenLane V1` for lane supervision
- `MapTR`-style vectorized priors for public map tokens

Private or proprietary dataset compatibility is intentionally out of scope in this public repository.

## Measured Deployment Notes

Measured RTX 5000 results are summarized in [benchmarks/rtx5000.md](./benchmarks/rtx5000.md).

- Full model, eager PyTorch, `256x704`, batch 1: mean `10.872 ms`, p95 `10.977 ms`
- Exportable core, TensorRT FP16-enabled engine, `256x704`, batch 1: mean `0.785 ms`, p95 `0.795 ms`

Those TensorRT numbers apply to the current exportable core only, not the full end-to-end multimodal pipeline.
