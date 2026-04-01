# Lane Dataset Status

This repo currently has two different public lane-data tracks:

- `OpenLane V1` is the native TSQBEV lane path today.
- `OpenLane-V2` is the reset-stack / future map-lane path.

Primary sources:

- OpenLane V1 data docs: <https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md>
- OpenLane-V2 getting started: <https://github.com/OpenDriveLab/OpenLane-V2/blob/main/docs/getting_started.md>
- OpenLane-V2 data manifest: <https://github.com/OpenDriveLab/OpenLane-V2/blob/main/data/README.md>
- PersFormer OpenLane reference: <https://github.com/OpenDriveLab/PersFormer_3DLane>

## What Works Today

The current repo code reads `OpenLane V1` directly through [src/tsqbev/datasets.py](../src/tsqbev/datasets.py).
The expected root is:

```text
<openlane-root>/
  images/
    training/
    validation/
  lane3d_300/
    training/
    validation/
  lane3d_1000/
    training/
    validation/
```

The smallest useful local baseline is:

- `images/training`
- `images/validation`
- `lane3d_300/training`
- `lane3d_300/validation`

That is enough to run:

```bash
uv run tsqbev train-openlane \
  --dataset-root /path/to/OpenLane \
  --subset lane3d_300 \
  --artifact-dir artifacts/baselines
```

## Current Download Reality

### OpenLane V1

OpenLane V1 is still form-gated in the official repo:

- official link target: <https://forms.gle/BzxxkUZDuPTqFKgu9>

So there is no reliable unattended public CLI download path in this repo yet. If you already have an
approved archive or mounted dataset root, TSQBEV can use it immediately.

### OpenLane-V2

OpenLane-V2 publishes official Google Drive and OpenDataLab links in its data manifest. This repo
now has a reproducible downloader for the official Google Drive archives:

```bash
uv run tsqbev list-openlanev2-archives

uv run tsqbev download-openlanev2 \
  --archive-key sample \
  --output-dir /home/achbogga/projects/research/openlanev2 \
  --extract-openlanev2
```

Supported archive keys currently include:

- `sample`
- `subset_a_info`
- `subset_a_info_ls`
- `subset_a_sdmap`
- `subset_b_info`

## Local Status On This Machine

As of the latest sync:

- official `OpenLane` repo checkout exists at `/home/achbogga/projects/OpenLane`
- official `OpenLane-V2` repo checkout exists at `/home/achbogga/projects/OpenLane-V2`
- official `OpenLane-V2` sample archive was downloaded and extracted under:
  `/home/achbogga/projects/research/openlanev2/data/OpenLane-V2`
- official `OpenLane-V2` `subset_a_info_ls` archive was downloaded and verified under:
  `/home/achbogga/projects/research/openlanev2/OpenLane-V2_subset_A_info-ls.tar.gz`
- official `OpenLane-V2` `subset_a_sdmap` archive was downloaded and extracted under:
  `/home/achbogga/projects/research/openlanev2/OpenLane-V2_subset_A_sdmap.tar`

The sample download was verified against the official md5:

- `OpenLane-V2_sample.tar`
- md5 `21c607fa5a1930275b7f1409b25042a0`

The larger `subset_a_info` archive is currently blocked on the public Google Drive mirror by
`Quota exceeded`. The repo downloader now reports that as a structured `html_error` with
`google_drive_quota_exceeded` instead of crashing.

## Research Interpretation

For immediate TSQBEV lane baselines, `OpenLane V1` remains the highest-ROI dataset because the repo
already reads and batches it directly.

For the dense-BEV reset path, `OpenLane-V2` is still valuable, but it is a separate integration
project rather than a drop-in replacement for the current `OpenLaneDataset`.
