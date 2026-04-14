#!/usr/bin/env bash
set -euo pipefail

BEVDET_ROOT="${BEVDET_ROOT:-/home/achbogga/projects/BEVDet}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/storage/research/nuscenes}"
BEVDET_IMAGE_TAG="${BEVDET_IMAGE_TAG:-tsqbev-bevdet-official:latest}"
CONFIG_REL="${CONFIG_REL:-configs/bevdet/bevdet-r50-cbgs.py}"
VERSION="${VERSION:-v1.0-mini}"
EXTRA_TAG="${EXTRA_TAG:-}"
TSQBEV_ROOT="${TSQBEV_ROOT:-/home/achbogga/projects/tsqbev-poc}"
RESEARCH_ASSETS_ROOT="${RESEARCH_ASSETS_ROOT:-/home/achbogga/projects/research_assets}"
CONTAINER_BEVDET_ROOT="${CONTAINER_BEVDET_ROOT:-/workspace/BEVDet}"
CONTAINER_DATASET_ROOT="${CONTAINER_DATASET_ROOT:-/dataset}"
CONTAINER_TSQBEV_ROOT="${CONTAINER_TSQBEV_ROOT:-/workspace/tsqbev-poc}"
CONTAINER_RESEARCH_ASSETS_ROOT="${CONTAINER_RESEARCH_ASSETS_ROOT:-/workspace/research_assets}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${TSQBEV_ROOT}/artifacts/public_student_bevdet}"
WORK_DIR="${WORK_DIR:-${ARTIFACT_DIR}/work_dir}"
SUMMARY_PATH="${SUMMARY_PATH:-${ARTIFACT_DIR}/bevdet_run_summary.json}"
EPOCHS="${EPOCHS:-6}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
LOAD_FROM="${LOAD_FROM:-}"
TRAIN_LOG="${TRAIN_LOG:-${ARTIFACT_DIR}/train.log}"
TEST_LOG="${TEST_LOG:-${ARTIFACT_DIR}/test.log}"

path_in_container() {
  local host_path="$1"
  case "${host_path}" in
    "${TSQBEV_ROOT}"/*)
      printf '%s%s' "${CONTAINER_TSQBEV_ROOT}" "${host_path#${TSQBEV_ROOT}}"
      ;;
    "${BEVDET_ROOT}"/*)
      printf '%s%s' "${CONTAINER_BEVDET_ROOT}" "${host_path#${BEVDET_ROOT}}"
      ;;
    "${DATASET_ROOT}"/*)
      printf '%s%s' "${CONTAINER_DATASET_ROOT}" "${host_path#${DATASET_ROOT}}"
      ;;
    "${RESEARCH_ASSETS_ROOT}"/*)
      printf '%s%s' "${CONTAINER_RESEARCH_ASSETS_ROOT}" "${host_path#${RESEARCH_ASSETS_ROOT}}"
      ;;
    *)
      printf '%s' "${host_path}"
      ;;
  esac
}

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
CONTAINER_WORK_DIR="$(path_in_container "${WORK_DIR}")"
CONTAINER_RESULT_PREFIX="$(path_in_container "${RESULT_PREFIX}")"
CONTAINER_RESULT_JSON="$(path_in_container "${RESULT_JSON}")"
CONTAINER_LOAD_FROM=""
if [[ -n "${LOAD_FROM}" ]]; then
  CONTAINER_LOAD_FROM="$(path_in_container "${LOAD_FROM}")"
fi

if [[ -z "${CFG_STRING:-}" ]]; then
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
    CFG_OPTIONS+=("load_from=${CONTAINER_LOAD_FROM}")
  fi

  CFG_STRING="${CFG_OPTIONS[*]}"
fi

if [[ -n "${LOAD_FROM}" && "${CFG_STRING}" != *"load_from="* ]]; then
  CFG_STRING="${CFG_STRING} load_from=${CONTAINER_LOAD_FROM}"
fi

set +e
docker run --rm --gpus all --shm-size 16g \
  -e CONFIG_REL="${CONFIG_REL}" \
  -e CFG_STRING="${CFG_STRING}" \
  -e CONTAINER_WORK_DIR="${CONTAINER_WORK_DIR}" \
  -v "${BEVDET_ROOT}:${CONTAINER_BEVDET_ROOT}" \
  -v "${DATASET_ROOT}:${CONTAINER_DATASET_ROOT}" \
  -v "${TSQBEV_ROOT}:${CONTAINER_TSQBEV_ROOT}" \
  -v "${RESEARCH_ASSETS_ROOT}:${CONTAINER_RESEARCH_ASSETS_ROOT}" \
  -w "${CONTAINER_BEVDET_ROOT}" \
  "${BEVDET_IMAGE_TAG}" \
  /bin/bash -lc "export SETUPTOOLS_USE_DISTUTILS=stdlib && set -euo pipefail && \
    mkdir -p /tmp/tsqbev_pyfix && \
    printf '%s\n' 'import distutils' 'from distutils import version as _version' \"if not hasattr(distutils, 'version'):\" '    distutils.version = _version' > /tmp/tsqbev_pyfix/sitecustomize.py && \
    export PYTHONPATH=/tmp/tsqbev_pyfix:\${PYTHONPATH:-} && \
    mkdir -p data && ln -sfn ${CONTAINER_DATASET_ROOT} data/nuscenes && \
    python -m pip install --no-cache-dir 'setuptools<60' >/tmp/bevdet_setuptools_fix.log 2>&1 && \
    python -m pip install --no-cache-dir 'numpy<1.24' >/tmp/bevdet_numpy_fix.log 2>&1 && \
    python -m pip install --no-cache-dir cumm-cu113 spconv-cu113 >/tmp/bevdet_spconv_fix.log 2>&1 && \
    python -m pip install --no-cache-dir 'yapf<0.40' >/tmp/bevdet_yapf_fix.log 2>&1 && \
    if ! python -m pip install --no-build-isolation -v -e . >/tmp/bevdet_editable_install.log 2>&1; then \
      cat /tmp/bevdet_editable_install.log; \
      exit 1; \
    fi && \
    python tools/train.py ${CONFIG_REL} --gpu-id 0 --work-dir ${CONTAINER_WORK_DIR} --validate --cfg-options ${CFG_STRING}" \
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
CONTAINER_CHECKPOINT_PATH="$(path_in_container "${CHECKPOINT_PATH}")"

set +e
docker run --rm --gpus all --shm-size 16g \
  -e CONFIG_REL="${CONFIG_REL}" \
  -e CFG_STRING="${CFG_STRING}" \
  -e CONTAINER_CHECKPOINT_PATH="${CONTAINER_CHECKPOINT_PATH}" \
  -e CONTAINER_RESULT_PREFIX="${CONTAINER_RESULT_PREFIX}" \
  -v "${BEVDET_ROOT}:${CONTAINER_BEVDET_ROOT}" \
  -v "${DATASET_ROOT}:${CONTAINER_DATASET_ROOT}" \
  -v "${TSQBEV_ROOT}:${CONTAINER_TSQBEV_ROOT}" \
  -v "${RESEARCH_ASSETS_ROOT}:${CONTAINER_RESEARCH_ASSETS_ROOT}" \
  -w "${CONTAINER_BEVDET_ROOT}" \
  "${BEVDET_IMAGE_TAG}" \
  /bin/bash -lc "export SETUPTOOLS_USE_DISTUTILS=stdlib && set -euo pipefail && \
    mkdir -p /tmp/tsqbev_pyfix && \
    printf '%s\n' 'import distutils' 'from distutils import version as _version' \"if not hasattr(distutils, 'version'):\" '    distutils.version = _version' > /tmp/tsqbev_pyfix/sitecustomize.py && \
    export PYTHONPATH=/tmp/tsqbev_pyfix:\${PYTHONPATH:-} && \
    mkdir -p data && ln -sfn ${CONTAINER_DATASET_ROOT} data/nuscenes && \
    python -m pip install --no-cache-dir 'setuptools<60' >/tmp/bevdet_setuptools_fix.log 2>&1 && \
    python -m pip install --no-cache-dir 'numpy<1.24' >/tmp/bevdet_numpy_fix.log 2>&1 && \
    python -m pip install --no-cache-dir cumm-cu113 spconv-cu113 >/tmp/bevdet_spconv_fix.log 2>&1 && \
    python -m pip install --no-cache-dir 'yapf<0.40' >/tmp/bevdet_yapf_fix.log 2>&1 && \
    if ! python -m pip install --no-build-isolation -v -e . >/tmp/bevdet_editable_install.log 2>&1; then \
      cat /tmp/bevdet_editable_install.log; \
      exit 1; \
    fi && \
    python tools/test.py ${CONFIG_REL} ${CONTAINER_CHECKPOINT_PATH} --gpu-id 0 --format-only \
      --eval-options jsonfile_prefix=${CONTAINER_RESULT_PREFIX} --cfg-options ${CFG_STRING}" \
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
