# nuScenes 32-Sample Overfit Gate

This note records the repaired exact-token overfit gate run for the promoted `v1.0-mini` recipe.

Grounding:

- overfit gate contract: [specs/005-scale-gate-contract.md](../../specs/005-scale-gate-contract.md)
- gate runner implementation: [src/tsqbev/overfit.py](../../src/tsqbev/overfit.py)
- official nuScenes metric stack: <https://github.com/nutonomy/nuscenes-devkit>

## Run Configuration

- dataset: `nuScenes v1.0-mini`
- subset size: `32` fixed sample tokens
- split used for both train and eval: `mini_train`
- preset: `rtx5000-nuscenes-query-boost`
- epochs: `128`
- max train steps: `1024`
- batch size: `4`
- grad accumulation: `1`
- device: local RTX 5000

Primary artifacts:

- `artifacts/gates/recovery_v1/overfit_gate/summary.json`
- `artifacts/gates/recovery_v1/overfit_gate/predictions_subset.json`

## Result

The gate failed.

| Metric | Value |
| --- | ---: |
| Initial train total | `73.2606` |
| Final train total | `38.8998` |
| Train-total ratio | `0.5310` |
| Final val total | `38.0125` |
| Official same-subset `mAP` | `5.3287e-04` |
| Official same-subset `NDS` | `8.5868e-03` |
| Nonzero classes | `3` |
| `car AP @ 4.0m` | `0.0` |
| RTX 5000 synthetic forward mean | `17.7696 ms` |
| Boxes per sample mean | `112.0` |
| Ego-frame translation `p99` | `54.58 m` |
| Ego-frame translation max | `61.16 m` |

The repaired run no longer sits fully in the all-zero regime, but the required `car AP @ 4.0m`
criterion was still not met.

## Gate Verdict

Required thresholds:

- train-total ratio `<= 0.40`
- same-subset `NDS >= 0.10`
- same-subset `mAP > 0.0`
- `car AP @ 4.0m > 0.0`

Observed verdict:

- train-total ratio: failed
- same-subset `NDS`: failed
- same-subset `mAP`: passed
- `car AP @ 4.0m`: failed

## Interpretation

This is a useful negative result, and it is materially better than the earlier collapsed export.

- The repaired student learns a real signal on the exact fixed subset: official same-subset
  `NDS` moved to `0.0085868`, `mAP` remained nonzero, and `3` classes reached nonzero AP.
- The export path is not geometrically collapsed in the ego frame. The apparent `1000m+`
  translation norms in the raw result JSON are nuScenes global coordinates, not ego-relative
  ranges, so they must not be used directly as a geometry sanity signal.
- The remaining failure is not “garbage world coordinates”; it is weak memorization plus too many
  surviving detections per sample.
- This failure still reinforces the scale gate. The correct next investment is teacher-guided
  supervision and tighter ranking/selection, not larger compute.

## Next Step

The strongest measured follow-up on the same fixed subset so far is the corrected
`replace_lidar` teacher-seeded recovery probe:

| Metric | Value |
| --- | ---: |
| Train-total ratio | `0.5617` |
| Same-subset official `NDS` | `0.0401210` |
| Same-subset official `mAP` | `0.0214468` |
| Nonzero classes | `2` |
| `car AP @ 4.0m` | `0.0` |
| RTX 5000 synthetic forward mean | `16.9225 ms` |

Primary artifact:

- `artifacts/gates/recovery_v2_teacher_seed/overfit_gate/summary.json`

This is the first overfit probe that produced a material same-subset lift, but it still did not
clear the gate because the subset did not overfit enough and `car AP @ 4.0m` remained zero.

The highest-ROI next experiment on the current branch is therefore the repaired recovery rerun:

1. rerun the same 32-sample gate with the selected-checkpoint path enabled
2. keep the preferred `teacher_seed_mode=replace_lidar` geometry bootstrap
3. switch the detection loss to focal-style hard negatives
4. sweep the bounded `score_threshold` / `top_k` grid and select by official same-subset metrics
