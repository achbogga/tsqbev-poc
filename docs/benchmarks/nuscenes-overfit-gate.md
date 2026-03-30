# nuScenes 32-Sample Overfit Gate

This note records the repaired exact-token overfit gate runs for the promoted `v1.0-mini` recovery
branch.

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

## Current Best Result

The gate still fails, but the current best repaired teacher-anchor run is materially stronger.

| Metric | Value |
| --- | ---: |
| Initial train total | `33.4916` |
| Selected train total | `15.8509` |
| Train-total ratio | `0.4703` |
| Selected val total | `15.5109` |
| Official same-subset `mAP` | `0.1391` |
| Official same-subset `NDS` | `0.1001` |
| Nonzero classes | `7` |
| `car AP @ 4.0m` | `0.5327` |
| RTX 5000 synthetic forward mean | `17.9174 ms` |
| Boxes per sample mean | `64.0` |
| Ego-frame translation `p99` | `54.09 m` |
| Ego-frame translation max | `70.85 m` |

Primary artifact:

- `artifacts/gates/recovery_v6_teacher_anchor_balanced/overfit_gate/summary.json`

## Gate Verdict

Required thresholds:

- train-total ratio `<= 0.40`
- same-subset `NDS >= 0.10`
- same-subset `mAP > 0.0`
- `car AP @ 4.0m > 0.0`

Observed verdict on the current best repaired run:

- train-total ratio: failed
- same-subset `NDS`: passed
- same-subset `mAP`: passed
- `car AP @ 4.0m`: passed

## Interpretation

This is no longer a weak near-zero result.

- The repaired student now learns a real multi-class signal on the exact fixed subset:
  official same-subset `NDS` crossed `0.10`, `mAP` crossed `0.13`, `7` classes reached nonzero AP,
  and `car AP @ 4.0m` is strongly nonzero.
- The export path remains geometrically sane in ego coordinates while box count is held at `64`.
- The active blocker is now much narrower than before: optimization/memorization on the fixed
  subset, not gross geometry failure or complete vehicle collapse.
- This still reinforces the scale gate, but it changes the next investment. The highest-ROI work is
  now capacity/optimization on the repaired teacher-anchor recipe, not another blind rescue of car
  emergence.

## Next Step

The next highest-ROI experiment is no longer “recover any car signal at all.” That part now works.

The next bounded step is to push the repaired teacher-anchor recipe over the remaining optimization
gate:

1. keep the corrected class-balanced teacher-anchor seed selection
2. preserve the bounded selected-checkpoint and calibration path
3. increase effective optimization capacity on the same fixed subset
4. only claim the scale gate is clear once `train_total_ratio <= 0.40` on the same official metric path
