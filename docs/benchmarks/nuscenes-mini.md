# nuScenes v1.0-mini Baseline

This note records the bounded local research loops and promoted `v1.0-mini` baselines executed on
March 27-29, 2026.

Grounding:

- nuScenes official devkit and split definitions: <https://github.com/nutonomy/nuscenes-devkit>
- repo research loop contract: [specs/004-research-loop-contract.md](../../specs/004-research-loop-contract.md)

## Sweep V1

All sweep runs used:

- dataset: `nuScenes v1.0-mini`
- train split: `mini_train`
- validation split: `mini_val`
- backbone: pretrained `torchvision` `MobileNetV3-Large`
- image size: `256x704`
- device: local RTX 5000

| Recipe | Batch | Grad Accum | Backbone | Val Total | Synthetic Mean ms | Decision |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `mini_mbv3_frozen_bs2` | 2 | 2 | frozen | 28.0525 | 15.6993 | keep |
| `mini_mbv3_frozen_bs4` | 4 | 1 | frozen | 29.4311 | 15.6731 | discard |
| `mini_mbv3_unfrozen_bs2` | 2 | 2 | unfrozen | 29.6107 | 15.6858 | discard |

Interpretation:

- increasing batch size used more throughput headroom but hurt `mini_val` loss after one epoch
- unfreezing the pretrained backbone also lost to the frozen baseline in this small sweep
- the best recipe stayed the conservative frozen-backbone configuration

The sweep artifacts are stored in:

- `artifacts/baselines/research_loop/results.jsonl`
- `artifacts/baselines/research_loop/results.tsv`
- `artifacts/baselines/research_loop/summary.json`

## Promoted 4-Epoch Mini Baseline (Historical)

The selected recipe was promoted to a longer baseline run:

- recipe: `mini_mbv3_frozen_bs2`
- epochs: `4`
- batch size: `2`
- grad accumulation: `2`
- learning rate: `3e-4`

Validation-loss trajectory:

| Epoch | Val Object Cls | Val Object Box | Val Total |
| --- | ---: | ---: | ---: |
| 1 | 3.3888 | 25.8781 | 29.2669 |
| 2 | 3.6466 | 23.1178 | 26.7644 |
| 3 | 3.3052 | 21.5022 | 24.8073 |
| 4 | 3.3129 | 21.0877 | 24.4006 |

Official `mini_val` evaluation after epoch 4:

| Metric | Value |
| --- | ---: |
| `mAP` | `1.5376e-05` |
| `NDS` | `7.6880e-06` |
| `car AP @ 4.0m` | `6.1504e-04` |
| `eval_time_s` | `0.9315` |

Observed status:

- this is a functional nonzero baseline, not a competitive detector yet
- only the `car` class reached nonzero AP in the official `mini_val` run
- the run is useful as a verified public starting point because the end-to-end training, checkpointing, export, and official evaluation path all completed successfully

Baseline artifacts:

- training history: `artifacts/baselines/mini_selected/nuscenes/history.json`
- checkpoint: `artifacts/baselines/mini_selected/nuscenes/checkpoint_last.pt`
- prediction export: `artifacts/baselines/mini_selected/nuscenes_predictions.json`
- official eval summary: `artifacts/baselines/mini_selected/eval/nuscenes/metrics_summary.json`

## Sweep V3: Incumbent-Guided Explore/Exploit

The second bounded loop changed the selection rule. Recipes are now ranked by:

1. official `mini_val` `NDS`
2. official `mini_val` `mAP`
3. validation loss only as a tiebreaker

That change mattered in practice: the lowest-loss recipe still had `NDS = 0.0`, while a
slightly higher-loss recipe produced the first clearly nonzero official `NDS`.

All V3 sweep runs used:

- dataset: `nuScenes v1.0-mini`
- train split: `mini_train`
- validation split: `mini_val`
- epochs: `6`
- export threshold: `0.05`
- device: local RTX 5000

| Recipe | Backbone | Query Budget | Val Total | mAP | NDS | Mean ms | Source Mix | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `mini_balanced_mbv3_frozen` | frozen `MobileNetV3-Large` | `96 / 64 / 32` | 22.0419 | 0.0 | 0.0 | 17.4492 | `50 / 33 / 17` | discard |
| `mini_propheavy_mbv3_frozen` | frozen `MobileNetV3-Large` | `64 / 96 / 32` | 23.0423 | `3.1083e-04` | `1.5541e-04` | 17.1797 | `33 / 50 / 17` | discard |
| `mini_propheavy_effb0_frozen` | frozen `EfficientNet-B0` | `64 / 96 / 32` | 24.1645 | 0.0 | 0.0 | 21.6682 | `33 / 50 / 17` | discard |
| `mini_propheavy_mbv3_frozen_query_boost` | frozen `MobileNetV3-Large` | `64 / 112 / 32` | 20.1352 | `0.0001114` | `0.0158` | 17.1938 | `31 / 54 / 15` | promote |
| `mini_propheavy_mbv3_frozen_lr_down` | frozen `MobileNetV3-Large` | `64 / 96 / 32` | 24.7614 | 0.0 | 0.0 | 17.1298 | `33 / 50 / 17` | discard |

Interpretation:

- the router/source-balance fix was necessary but not sufficient
- simply lowering validation loss is still not enough; official detection metrics can remain zero
- the first exploit-stage `query_boost` mutation materially beat the initial explore-stage winner
- the promoted recipe now has a clearly nonzero official `NDS` and nonzero official `mAP`
- the direction is learnable without an external teacher, but it is still far below the scale gate

V3 artifacts:

- sweep ledger: `artifacts/research_v3/research_loop/results.jsonl`
- sweep ledger TSV: `artifacts/research_v3/research_loop/results.tsv`
- sweep summary: `artifacts/research_v3/research_loop/summary.json`

## Scale Decision

Current answer: do **not** scale by 10x compute yet.

The reasons are straightforward:

- best official `mini_val` `NDS` is still only `0.0158`
- best official `mini_val` `mAP` is still only `1.1140e-04`
- the model has not yet cleared a deliberate tiny-subset overfit gate
- a pretrained external LiDAR teacher has not yet produced a measured lift artifact

The formal go/no-go rules are in [docs/scaling-gates.md](../scaling-gates.md) and
[specs/005-scale-gate-contract.md](../../specs/005-scale-gate-contract.md).
