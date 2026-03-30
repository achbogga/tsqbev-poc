# Spec 005: Scale Gate Contract

## Goal

Prevent premature 10x-compute scale-up by requiring strong, measured evidence that the current
direction is both learnable and improving for the right reasons.

## Current Measured Status

As of the bounded `nuScenes v1.0-mini` sweep recorded under `artifacts/research_v3/`:

- best recipe: `mini_propheavy_mbv3_frozen_query_boost`
- official `mini_val` `NDS`: `0.0158068933`
- official `mini_val` `mAP`: `0.0001114034`
- final validation total: `20.1352`
- source mix: `31.25% LiDAR / 53.57% proposal / 15.18% global`
- synthetic RTX 5000 forward latency: `17.19 ms` mean

This is evidence that the current direction is alive. It is not enough evidence to justify a 10x
compute scale-up.

No passing artifacts are currently recorded for:

- Gate 2: small-subset overfit
- Gate 3A: geometry sanity
- Gate 4: teacher-lift
- Gate 6: repeatability

## Required Gates Before 10x Compute

### Gate 0: Repo Integrity

All of the following must pass on the exact code/config branch that will be promoted:

- `ruff`
- `mypy`
- `pytest`
- ONNX export smoke
- TensorRT smoke for the exportable core
- bounded research loop completes without runtime errors

### Gate 1: Source-Mix Stability

The routed query bank must remain genuinely multimodal.

Minimum criteria:

- no single source may exceed `80%` of the selected query bank on the monitored validation batches
- proposal-source share must remain at least `20%`
- LiDAR-source share must remain at least `20%`
- the measured source mix must be stable across at least `8` validation batches

If the query bank collapses back to one source, scaling is blocked.

### Gate 2: Small-Subset Overfit

The model must be able to fit a deliberately small public subset before larger training is funded.

Required protocol:

- train on a fixed `32`-sample subset
- evaluate on the same `32` samples
- use the exact architecture that is a candidate for promotion

Minimum criteria:

- final train total is at most `40%` of the initial train total
- official same-subset `NDS >= 0.10`
- official same-subset `mAP > 0.0`
- at least one class reaches nonzero AP at the `4.0 m` threshold

If the model cannot overfit a tiny subset, larger-scale training is blocked.

### Gate 3: Mini Generalization

The promoted public mini recipe must clear all of the following on `mini_train / mini_val`:

- official `NDS >= 0.05`
- official `mAP >= 0.01`
- at least `3` classes with nonzero mean distance AP
- `car AP @ 4.0m >= 0.05`
- translation error `mATE < 1.0`

The intent is not to claim competitiveness. The intent is to reject directions that still only
look good through surrogate losses.

### Gate 3A: Geometry Sanity

The promoted mini recipe must not rely on pathological exported predictions to achieve its score.

Minimum criteria:

- exported boxes per sample mean `<= 40`
- exported boxes per sample `p95 <= 60`
- exported ego-frame translation norm `p99 <= 120 m`
- exported ego-frame translation norm max `<= 150 m`

If exported predictions still explode in count or range, scaling is blocked even if official
metrics are nonzero.

### Gate 4: Teacher-Lift

If a pretrained external LiDAR teacher is introduced, it must produce a material improvement over
the student-only baseline at the same public mini setting.

Minimum criteria:

- teacher-cache audit coverage `>= 95%` on both `mini_train` and `mini_val`
- absolute `NDS` lift of at least `+0.02`, or
- relative `NDS` lift of at least `2x`

The lift must be measured on official `mini_val` evaluation, not only on loss curves.

### Gate 5: Efficiency Discipline

The recipe selected for larger training must still respect the deployment direction.

Minimum criteria:

- synthetic RTX 5000 forward mean latency `<= 25 ms`
- ONNX export smoke still passes
- exportable-core TensorRT path still builds

If a recipe improves quality but breaks the deployment path, it is not a default promotion target.

### Gate 6: Repeatability

Before 10x compute is authorized, the promoted recipe must be rerun at least twice.

Minimum criteria:

- rerun `NDS` variation stays within `10%` relative of the mean
- rerun source-mix diagnostics stay within `10%` relative of the chosen run
- no run may regress to `mAP == 0.0`

## Promotion Rule

10x compute is authorized only if every gate above passes.

If any gate fails, the next investment must go into architecture, teacher bootstrapping,
supervision, or exportability rather than raw scale.
