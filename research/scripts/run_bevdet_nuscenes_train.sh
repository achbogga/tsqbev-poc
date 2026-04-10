#!/usr/bin/env bash
set -euo pipefail

BEVDET_ROOT="${BEVDET_ROOT:-/home/achbogga/projects/BEVDet}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/storage/research/nuscenes}"
BEVDET_IMAGE_TAG="${BEVDET_IMAGE_TAG:-tsqbev-bevdet-official:latest}"
CONFIG_REL="${CONFIG_REL:-configs/bevdet/bevdet-r50-cbgs.py}"
VERSION="${VERSION:-v1.0-mini}"
EXTRA_TAG="${EXTRA_TAG:-}"
TSQBEV_ROOT="${TSQBEV_ROOT:-/home/achbogga/projects/tsqbev-poc}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${TSQBEV_ROOT}/artifacts/public_student_bevdet}"
WORK_DIR="${WORK_DIR:-${ARTIFACT_DIR}/work_dir}"
SUMMARY_PATH="${SUMMARY_PATH:-${ARTIFACT_DIR}/bevdet_run_summary.json}"
EPOCHS="${EPOCHS:-6}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
LOAD_FROM="${LOAD_FROM:-}"
TRAIN_LOG="${TRAIN_LOG:-${ARTIFACT_DIR}/train.log}"
TEST_LOG="${TEST_LOG:-${ARTIFACT_DIR}/test.log}"

if [[ -z "${EXTRA_TAG}" ]]; then
  if [[ "${VERSION}" == "v1.0-mini" ]]; then
    EXTRA_TAG="bevdetv3-mini"
  else
    EXTRA_TAG="bevdetv3-nuscenes"
  fi
fi

mkdir -p "${ARTIFACT_DIR}" "${WORK_DIR}"
RESULT_PREFIX="${WORK_DIR}/format_only"
RESULT_JSON="${RESULT_PREFIX}/pts_bbox/results_nusc.json"

CFG_OPTIONS=(
  "data.samples_per_gpu=${SAMPLES_PER_GPU}"
  "data.workers_per_gpu=${WORKERS_PER_GPU}"
  "data.train.dataset.data_root=data/nuscenes/"
  "data.train.dataset.ann_file=data/nuscenes/${EXTRA_TAG}_infos_train.pkl"
  "data.val.data_root=data/nuscenes/"
  "data.val.ann_file=data/nuscenes/${EXTRA_TAG}_infos_val.pkl"
  "data.test.data_root=data/nuscenes/"
  "data.test.ann_file=data/nuscenes/${EXTRA_TAG}_infos_val.pkl"
  "runner.max_epochs=${EPOCHS}"
  "evaluation.interval=1"
  "checkpoint_config.interval=1"
  "log_config.interval=20"
)

if [[ -n "${LOAD_FROM}" ]]; then
  CFG_OPTIONS+=("load_from=${LOAD_FROM}")
fi

CFG_STRING="${CFG_OPTIONS[*]}"

set +e
docker run --rm --gpus all --shm-size 16g \
  -v "${BEVDET_ROOT}:/workspace/BEVDet" \
  -v "${DATASET_ROOT}:/dataset" \
  -v "${TSQBEV_ROOT}:/workspace/tsqbev-poc" \
  -w /workspace/BEVDet \
  "${BEVDET_IMAGE_TAG}" \
  /bin/bash -lc "set -euo pipefail && \
    mkdir -p data && ln -sfn /dataset data/nuscenes && \
    python -m pip install -v -e . >/tmp/bevdet_editable_install.log 2>&1 || cat /tmp/bevdet_editable_install.log && \
    python tools/train.py ${CONFIG_REL} --gpu-id 0 --work-dir ${WORK_DIR} --validate --cfg-options ${CFG_STRING}" \
  2>&1 | tee "${TRAIN_LOG}"
train_status=${PIPESTATUS[0]}
set -e

if [[ ${train_status} -ne 0 ]]; then
  exit "${train_status}"
fi

CHECKPOINT_PATH="${WORK_DIR}/latest.pth"
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(ls -1 "${WORK_DIR}"/epoch_*.pth | sort | tail -n 1)"
fi

set +e
docker run --rm --gpus all --shm-size 16g \
  -v "${BEVDET_ROOT}:/workspace/BEVDet" \
  -v "${DATASET_ROOT}:/dataset" \
  -v "${TSQBEV_ROOT}:/workspace/tsqbev-poc" \
  -w /workspace/BEVDet \
  "${BEVDET_IMAGE_TAG}" \
  /bin/bash -lc "set -euo pipefail && \
    mkdir -p data && ln -sfn /dataset data/nuscenes && \
    python -m pip install -v -e . >/tmp/bevdet_editable_install.log 2>&1 || cat /tmp/bevdet_editable_install.log && \
    python tools/test.py ${CONFIG_REL} ${CHECKPOINT_PATH} --gpu-id 0 --format-only \
      --eval-options jsonfile_prefix=${RESULT_PREFIX} --cfg-options ${CFG_STRING}" \
  2>&1 | tee "${TEST_LOG}"
test_status=${PIPESTATUS[0]}
set -e

if [[ ${test_status} -ne 0 ]]; then
  exit "${test_status}"
fi

python3 - "${SUMMARY_PATH}" "${CHECKPOINT_PATH}" "${RESULT_JSON}" "${TRAIN_LOG}" "${TEST_LOG}" <<'PY'
from pathlib import Path
import json
import sys

summary_path = Path(sys.argv[1])
payload = {
    "checkpoint_path": str(Path(sys.argv[2])),
    "prediction_path": str(Path(sys.argv[3])),
    "train_log": str(Path(sys.argv[4])),
    "test_log": str(Path(sys.argv[5])),
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
