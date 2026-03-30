# OpenPCDet CenterPoint-PointPillar on nuScenes v1.0-mini

This benchmark records the first completed external LiDAR-teacher bootstrap for `tsqbev-poc`.

Primary sources:

- OpenPCDet README model zoo: <https://github.com/open-mmlab/OpenPCDet>
- OpenPCDet install guide: <https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md>
- OpenPCDet getting started: <https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md>
- OpenPCDet nuScenes config:
  <https://github.com/open-mmlab/OpenPCDet/blob/master/tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml>

## Verified Local Setup

External repo root:

- `/home/achbogga/projects/OpenPCDet_official`

Teacher checkpoint:

- `/home/achbogga/projects/OpenPCDet_official/checkpoints/cbgs_dyn_pp_centerpoint.pth`

Local CUDA toolkit used for the build:

- `/home/achbogga/projects/OpenPCDet_official/.cuda-12.6/usr/local/cuda-12.6`

Local compatibility shim used for current NumPy:

- `/home/achbogga/projects/OpenPCDet_official/compat/sitecustomize.py`

The OpenPCDet CUDA extensions were built successfully on this workstation and import cleanly with:

- `torch 2.7.1+cu126`
- `spconv 2.3.8`
- `torch_scatter 2.1.2+pt27cu126`

## Measured Teacher Eval

Command family used:

```bash
cd /home/achbogga/projects/OpenPCDet_official
. .venv/bin/activate
export PYTHONPATH=$PWD/compat:$PWD:$PYTHONPATH
export CUDA_HOME=$PWD/.cuda-12.6/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

python tools/test.py \
  --cfg_file tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml \
  --ckpt checkpoints/cbgs_dyn_pp_centerpoint.pth \
  --workers 4 \
  --batch_size 1 \
  --save_to_file \
  --eval_tag mini_teacher_probe \
  --set \
    DATA_CONFIG.VERSION v1.0-mini \
    DATA_CONFIG.DATA_PATH /home/achbogga/projects/OpenPCDet_official/data/nuscenes
```

Measured result on `v1.0-mini` `mini_val`:

| Model | Split | mAP | NDS | sec/example |
| --- | --- | ---: | ---: | ---: |
| OpenPCDet `CenterPoint-PointPillar` | `v1.0-mini` `mini_val` | 0.4369 | 0.4997 | 0.1027 |

Result artifact root:

- `/home/achbogga/projects/OpenPCDet_official/output/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint/default/eval/epoch_no_number/val/mini_teacher_probe`

Key exported teacher JSON:

- `/home/achbogga/projects/OpenPCDet_official/output/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint/default/eval/epoch_no_number/val/mini_teacher_probe/final_result/data/results_nusc.json`

## Cache Conversion

Converted into repo-native teacher targets with:

```bash
cd /home/achbogga/projects/tsqbev-poc
uv run tsqbev cache-teacher-nuscenes \
  --dataset-root /home/achbogga/projects/research/nuscenes \
  --version v1.0-mini \
  --result-json /home/achbogga/projects/OpenPCDet_official/output/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint/default/eval/epoch_no_number/val/mini_teacher_probe/final_result/data/results_nusc.json \
  --teacher-cache-dir artifacts/teacher_cache/centerpoint_pointpillar_mini \
  --top-k 300
```

Measured cache result for the official `mini_val` teacher output:

- stored records: `81`
- top-k per sample: `300`

Cache summary:

- [summary.json](/home/achbogga/projects/tsqbev-poc/artifacts/teacher_cache/centerpoint_pointpillar_mini/summary.json)

## mini_train Export And Import

The stock OpenPCDet `tools/test.py` path always evaluates `v1.0-mini` against `mini_val`, so the
`mini_train` teacher export was done with an export-only script that overrides
`INFO_PATH['test'] = ['nuscenes_infos_10sweeps_train.pkl']` and never calls `dataset.evaluation()`.

Measured `mini_train` export result:

- samples exported: `323`
- average predicted objects per sample: `164.66`
- result JSON:
  `/home/achbogga/projects/OpenPCDet_official/output/mini_train_export_true/results_nusc.json`

Imported into the same repo-native teacher cache with:

```bash
cd /home/achbogga/projects/tsqbev-poc
uv run tsqbev cache-teacher-nuscenes \
  --dataset-root /home/achbogga/projects/research/nuscenes \
  --version v1.0-mini \
  --result-json /home/achbogga/projects/OpenPCDet_official/output/mini_train_export_true/results_nusc.json \
  --teacher-cache-dir artifacts/teacher_cache/centerpoint_pointpillar_mini \
  --top-k 300
```

## Coverage Audit

Audit commands:

```bash
cd /home/achbogga/projects/tsqbev-poc
uv run tsqbev audit-teacher-cache-nuscenes \
  --dataset-root /home/achbogga/projects/research/nuscenes \
  --version v1.0-mini \
  --teacher-cache-dir artifacts/teacher_cache/centerpoint_pointpillar_mini \
  --output-dir artifacts/teacher_cache_audit_train

uv run tsqbev audit-teacher-cache-nuscenes \
  --dataset-root /home/achbogga/projects/research/nuscenes \
  --version v1.0-mini \
  --split mini_val \
  --teacher-cache-dir artifacts/teacher_cache/centerpoint_pointpillar_mini \
  --output-dir artifacts/teacher_cache_audit_val
```

Measured audit results:

- `mini_train`: total `323`, present `323`, missing `0`, coverage `1.0`
- total samples: `81`
- present records: `81`
- missing records: `0`
- coverage: `1.0`

Audit summary:

- [summary.json](/home/achbogga/projects/tsqbev-poc/artifacts/teacher_cache_audit_train/summary.json)
- [summary.json](/home/achbogga/projects/tsqbev-poc/artifacts/teacher_cache_audit_val/summary.json)

## Interpretation

- The workstation is no longer blocked for the external OpenPCDet teacher path.
- The public pretrained teacher is strong enough on `v1.0-mini` to be a credible geometric bootstrap.
- The repo-native teacher cache contract is now exercised end to end with full `mini_train` and
  `mini_val` coverage.
- This is still not a teacher-lift claim for the student. The required next evidence is the paired
  teacher-on versus teacher-off `tsqbev` run now writing to
  `/home/achbogga/projects/tsqbev-poc/artifacts/research_teacher_v1/`.
