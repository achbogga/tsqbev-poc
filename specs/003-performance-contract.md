# Spec 003: Performance Contract

## Target Hardware

- primary production target: 1 NVIDIA Orin

## Latency Gates

- production: `p95 <= 100 ms` in FP16 TensorRT
- stretch: `p95 <= 50 ms` in INT8 TensorRT

## Predictor

Use a simple gate:

`pred_ms = b0 + b1*params_M + b2*sample_ops_M + b3*lidar_pillars_K + b4*T + b5*activations_MB`

This predictor is not ground truth.
Measured latency is the real acceptance criterion.

## Policy

- no configuration that obviously misses the chosen tier should be trained first
- bounded local research is allowed only after the repo is functional
- the active research dataset contract is `nuScenes v1.0-mini`
- the loop must not promote a config without recorded validation evidence

## References

- HotBEV deployment and Orin-minded operator discipline: <https://proceedings.neurips.cc/paper_files/paper/2023/file/081b08068e4733ae3e7ad019fe8d172f-Paper-Conference.pdf>
