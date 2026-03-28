# Scaling Gates

This note makes the current scale decision explicit.

## Current Answer

Do **not** scale `tsqbev-poc` by 10x compute yet.

The best bounded `nuScenes v1.0-mini` result so far is:

- recipe: proposal-heavy frozen `EfficientNet-B0`
- official `mini_val` `NDS`: `0.0127118571`
- official `mini_val` `mAP`: `0.0`
- final validation total: `23.6836`
- source mix: `33.3% LiDAR / 50.0% proposal / 16.7% global`
- synthetic RTX 5000 forward mean: `21.66 ms`

This is a meaningful improvement over the earlier collapsed-router state, but it is still below
the threshold for responsible scale-up.

## Why Not Yet

- the first nonzero official `NDS` is encouraging, but `mAP` is still exactly `0.0`
- only one recipe has escaped the all-zero official-metric regime
- the current direction has not yet demonstrated tiny-subset overfit
- there is not yet a pretrained external LiDAR teacher lift measured against the same mini setup

## What Must Happen First

1. Prove the model can overfit a fixed tiny public subset.
2. Push `mini_val` to a clearly nontrivial regime, not just a barely nonzero score.
3. Add a pretrained LiDAR teacher and demand a material official-metric lift.
4. Preserve the deployment path while improving quality.
5. Repeat the best recipe and verify stability.

The formal contract is in [specs/005-scale-gate-contract.md](../specs/005-scale-gate-contract.md).
