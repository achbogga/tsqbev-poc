# Scaling Gates

This note makes the current scale decision explicit.

The bounded research loop now also emits the same decision as a machine-readable
`scale_gate_verdict` block inside `research_loop/summary.json`.

## Current Answer

Do **not** scale `tsqbev-poc` by 10x compute yet.

The best bounded `nuScenes v1.0-mini` result so far is:

- recipe: `mini_propheavy_mbv3_frozen_query_boost`
- official `mini_val` `NDS`: `0.0158068933`
- official `mini_val` `mAP`: `0.0001114034`
- final validation total: `20.1352`
- source mix: `31.25% LiDAR / 53.57% proposal / 15.18% global`
- synthetic RTX 5000 forward mean: `17.19 ms`

This is a meaningful improvement over the earlier collapsed-router state, but it is still below
the threshold for responsible scale-up.

The latest measured 32-sample overfit gate also failed:

- train-total ratio: `0.5079`
- same-subset official `NDS`: `0.0003752`
- same-subset official `mAP`: `0.0007504`
- only `1` class was nonzero, and `car AP @ 4.0m` remained `0.0`
- artifact: `artifacts/gates/overfit_gate/summary.json`

## Why Not Yet

- the current best run has only a barely nonzero official `mAP`, so the detector is still far from
  a robust public baseline
- the repo now measures prediction-geometry sanity explicitly in the ego frame, and pathological
  exported box counts or ego-range values block promotion
- only one completed promoted recipe has clearly escaped the all-zero official-metric regime
- the current direction has now been tested on the tiny-subset overfit gate and failed it
- there is not yet a pretrained external LiDAR teacher lift measured against the same mini setup
- repeatability of the current best public recipe has not yet been demonstrated

## What Must Happen First

1. Prove the model can overfit a fixed tiny public subset.
2. Push `mini_val` to a clearly nontrivial regime, not just a barely nonzero score.
3. Add a pretrained LiDAR teacher and demand a material official-metric lift.
4. Preserve the deployment path while improving quality.
5. Repeat the best recipe and verify stability.

The repo now has a dedicated artifact path for Gate 2:

- `artifacts/gates/overfit_gate/summary.json`
- `docs/benchmarks/nuscenes-overfit-gate.md`

The teacher-cache audit prerequisite is now cleared, and the audited summaries are:

- `artifacts/teacher_cache_audit_train/summary.json`
- `artifacts/teacher_cache_audit_val/summary.json`

The formal contract is in [specs/005-scale-gate-contract.md](../specs/005-scale-gate-contract.md).
