# OpenPCDet CenterPoint Teacher Runbook

This is the fastest evidence-backed path to bootstrap a stronger LiDAR teacher without making the
core `tsqbev-poc` runtime depend on heavy 3D detection frameworks.

Primary sources:

- OpenPCDet README model zoo: <https://github.com/open-mmlab/OpenPCDet>
- OpenPCDet install guide: <https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md>
- OpenPCDet getting started: <https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md>
- OpenPCDet test entrypoint: <https://github.com/open-mmlab/OpenPCDet/blob/master/tools/test.py>
- OpenPCDet nuScenes evaluation/export path:
  <https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/nuscenes/nuscenes_dataset.py>

Local verification note:

- the runbook below was checked against official OpenPCDet commit
  `233f849829b6ac19afb8af8837a0246890908755` on March 29, 2026

## Chosen First Teacher

Use the official OpenPCDet `CenterPoint-PointPillar` nuScenes baseline first:

- config:
  `tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml`
- model-zoo row:
  `CenterPoint-PointPillar`
- official reported nuScenes validation score in the OpenPCDet README:
  `mAP 50.03`, `NDS 60.70`

This is the best first teacher because it is strong, public, and close to the student's existing
pillar-style LiDAR seed path.

## Separate Environment

Keep OpenPCDet in its own environment.

Do not add OpenPCDet, `spconv`, or other heavy sparse-3D dependencies to the default
`tsqbev-poc` environment.

## Current Local Status

This runbook has now been executed end to end on the current workstation.

Measured local facts:

- the OpenPCDet CUDA extensions built successfully against a local official CUDA `12.6` toolkit
- the official pretrained `CenterPoint-PointPillar` checkpoint ran on `nuScenes v1.0-mini`
- the resulting standard nuScenes JSON was converted into repo-local teacher-cache records
- the resulting `mini_train` teacher-cache coverage is `323 / 323 = 1.0`
- the resulting `mini_val` teacher-cache coverage is `81 / 81 = 1.0`

The external teacher benchmark and cache audit are recorded in
[`docs/benchmarks/openpcdet-centerpoint-mini.md`](benchmarks/openpcdet-centerpoint-mini.md).

The repo-local readiness check still exists and remains useful before trying this path on a new
machine:

```bash
uv run tsqbev check-openpcdet-env \
  --openpcdet-repo-root /home/achbogga/projects/OpenPCDet_official
```

## Dataset Layout

OpenPCDet expects a repo-local data root at `data/nuscenes`, with the actual nuScenes tree inside
it. The official getting-started doc shows:

```text
OpenPCDet/
  data/
    nuscenes/
      samples/
      sweeps/
      maps/
      v1.0-mini/
      v1.0-trainval/
```

The simplest non-copy path is a symlink:

```bash
cd /path/to/OpenPCDet
mkdir -p data
ln -s /home/achbogga/projects/research/nuscenes data/nuscenes
```

## Prepare nuScenes Mini Infos

The official getting-started doc uses the following command for nuScenes info generation, and the
same entrypoint also accepts `--version v1.0-mini`:

```bash
cd /path/to/OpenPCDet
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
  --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
  --version v1.0-mini
```

This writes the standard OpenPCDet info files under `data/nuscenes/v1.0-mini/`, including:

- `nuscenes_infos_10sweeps_train.pkl`
- `nuscenes_infos_10sweeps_val.pkl`

## Run the Pretrained Teacher on v1.0-mini

The official test entrypoint is `tools/test.py`. For the chosen teacher:

```bash
cd /path/to/OpenPCDet
. .venv/bin/activate
export PYTHONPATH=$PWD/compat:$PWD:$PYTHONPATH
export CUDA_HOME=$PWD/.cuda-12.6/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

python tools/test.py \
  --cfg_file tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml \
  --ckpt /path/to/centerpoint_pointpillar_checkpoint.pth \
  --batch_size 1 \
  --workers 4 \
  --save_to_file \
  --eval_tag mini_teacher_probe \
  --set \
    DATA_CONFIG.VERSION v1.0-mini \
    DATA_CONFIG.DATA_PATH /path/to/OpenPCDet/data/nuscenes
```

Notes grounded in the official code:

- `tools/test.py` loads the config and writes outputs under
  `output/<exp_group>/<tag>/<extra_tag>/eval/...`
- `pcdet/datasets/nuscenes/nuscenes_dataset.py` writes the standard nuScenes submission file as
  `results_nusc.json`
- the default `cfgs/dataset_configs/nuscenes_dataset.yaml` still labels the test split as `val`,
  so the output directory name is usually still `val` even when `DATA_CONFIG.VERSION` is
  `v1.0-mini`
- with the current latest Python package stack, a small NumPy compatibility shim is required for
  legacy aliases such as `np.int`; on this workstation that shim lives at
  `/home/achbogga/projects/OpenPCDet_official/compat/sitecustomize.py`
- when launching from the repo root, `DATA_CONFIG.DATA_PATH` should be set explicitly to the repo's
  `data/nuscenes` path so the loader does not resolve `../data/nuscenes` against the wrong cwd
- `results_nusc.json` is written by the evaluation path itself; `--save_to_file` is still useful
  because it also preserves the per-sample prediction dumps under `final_result/data/`

Expected result path pattern:

```text
OpenPCDet/output/nuscenes_models/cbgs_dyn_pp_centerpoint/default/eval/epoch_<id>/val/default/final_result/data/results_nusc.json
```

Verified local result on March 30, 2026:

- `mAP = 0.4369`
- `NDS = 0.4997`
- `Generate label finished(sec_per_example: 0.1027 second)`
- artifact root:
  `/home/achbogga/projects/OpenPCDet_official/output/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint/default/eval/epoch_no_number/val/mini_teacher_probe`

## Convert the Teacher Output Into tsqbev Cache Records

Use the repo-local conversion command:

```bash
cd /home/achbogga/projects/tsqbev-poc
uv run tsqbev cache-teacher-nuscenes \
  --dataset-root /home/achbogga/projects/research/nuscenes \
  --version v1.0-mini \
  --result-json /path/to/OpenPCDet/output/.../results_nusc.json \
  --teacher-cache-dir artifacts/teacher_cache/centerpoint_pointpillar_mini
```

This converts the standard nuScenes detection JSON into `TeacherCacheStore` records keyed by
sample token.

Verified local cache result:

- cache dir: `/home/achbogga/projects/tsqbev-poc/artifacts/teacher_cache/centerpoint_pointpillar_mini`
- stored records after `mini_train` and `mini_val` import: `404`

## Audit Coverage Before Any Teacher-Lift Claim

Do not treat a partial cache as a teacher experiment. Audit it first:

```bash
cd /home/achbogga/projects/tsqbev-poc
uv run tsqbev audit-teacher-cache-nuscenes \
  --dataset-root /home/achbogga/projects/research/nuscenes \
  --version v1.0-mini \
  --split mini_train \
  --teacher-cache-dir artifacts/teacher_cache/centerpoint_pointpillar_mini \
  --output-dir artifacts/teacher_cache/centerpoint_pointpillar_mini_audit_train

uv run tsqbev audit-teacher-cache-nuscenes \
  --dataset-root /home/achbogga/projects/research/nuscenes \
  --version v1.0-mini \
  --split mini_val \
  --teacher-cache-dir artifacts/teacher_cache/centerpoint_pointpillar_mini \
  --output-dir artifacts/teacher_cache/centerpoint_pointpillar_mini_audit_val
```

Required discipline for a real teacher-lift experiment:

- `mini_train` coverage should be at least `95%`
- `mini_val` coverage should be at least `95%`
- the teacher-lift run must be paired against a student-only run on the same mini setup

Verified local audit results:

- `mini_train`: present `323`, missing `0`, coverage `1.0`
- `mini_val`: present `81`, missing `0`, coverage `1.0`

Important split-selection note from the official OpenPCDet code:

- `tools/test.py` always builds the nuScenes dataloader with `training=False`
- in that mode, `NuScenesDataset` selects `INFO_PATH['test']`, not `DATA_SPLIT['test']`
- exporting real `mini_train` predictions therefore requires overriding `INFO_PATH['test']` to
  `['nuscenes_infos_10sweeps_train.pkl']`
- using `DATA_SPLIT.test=mini_train` alone does not switch the actual samples

## Train the Student From the Cache

```bash
cd /home/achbogga/projects/tsqbev-poc
uv run tsqbev train-nuscenes \
  --dataset-root /home/achbogga/projects/research/nuscenes \
  --artifact-dir artifacts/teacher_bootstrap \
  --preset rtx5000-nuscenes-teacher \
  --version v1.0-mini \
  --train-split mini_train \
  --split mini_val \
  --epochs 6 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --teacher-kind cache \
  --teacher-cache-dir artifacts/teacher_cache/centerpoint_pointpillar_mini
```

This keeps the heavy teacher outside the default repo runtime while still allowing a clean,
measured teacher-seed ablation inside `tsqbev-poc`.

The first paired teacher-on versus teacher-off bounded mini loop is now running with:

```bash
cd /home/achbogga/projects/tsqbev-poc
uv run tsqbev research-loop \
  --dataset-root /home/achbogga/projects/research/nuscenes \
  --artifact-dir artifacts/research_teacher_v1 \
  --teacher-kind cache \
  --teacher-cache-dir artifacts/teacher_cache/centerpoint_pointpillar_mini \
  --device cuda
```
