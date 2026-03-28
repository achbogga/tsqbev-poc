# nuScenes v1.0-mini Baseline

This note records the bounded local research loop and the promoted `v1.0-mini` baseline run executed on March 27, 2026.

Grounding:

- nuScenes official devkit and split definitions: <https://github.com/nutonomy/nuscenes-devkit>
- repo research loop contract: [specs/004-research-loop-contract.md](../../specs/004-research-loop-contract.md)

## Bounded Research Sweep

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
- `artifacts/baselines/research_loop/summary.json`

## Promoted 4-Epoch Mini Baseline

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
