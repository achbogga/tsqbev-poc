# nuScenes 32-Sample Overfit Gate

This note records the first exact-token overfit gate run for the promoted `v1.0-mini` recipe.

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

Primary artifact:

- `artifacts/gates/overfit_gate/summary.json`

## Result

The gate failed.

| Metric | Value |
| --- | ---: |
| Initial train total | `41.8276` |
| Final train total | `21.2427` |
| Train-total ratio | `0.5079` |
| Final val total | `21.3688` |
| Official same-subset `mAP` | `7.5044e-04` |
| Official same-subset `NDS` | `3.7522e-04` |
| Nonzero classes | `1` |
| `car AP @ 4.0m` | `0.0` |
| RTX 5000 synthetic forward mean | `17.3566 ms` |

The only nonzero AP came from `barrier` at the `4.0 m` threshold. The required `car AP @ 4.0m`
criterion was not met.

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

This is a useful negative result.

- The architecture does learn something on the exact fixed subset: train loss fell from `41.8276`
  to `21.2427`, and official same-subset `mAP` is nonzero.
- The current student still does not have enough capacity, supervision, or geometry quality to
  deliberately overfit the promoted subset under the official metric stack.
- This failure reinforces the existing scale gate. The correct next investment is not larger
  compute. It is stronger supervision or a stronger geometry prior, starting with the external
  teacher path.

## Next Step

The highest-ROI next experiment is a paired teacher bootstrap:

1. generate a public `CenterPoint-PointPillar` teacher cache on `v1.0-mini`
2. audit cache coverage on `mini_train` and `mini_val`
3. rerun the promoted mini recipe with `teacher_seed_mode=replace_lidar`
4. compare teacher-on vs teacher-off on official `mini_val`
