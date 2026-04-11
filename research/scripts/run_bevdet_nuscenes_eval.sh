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
ARTIFACT_DIR="${ARTIFACT_DIR:-${TSQBEV_ROOT}/artifacts/public_student_bevdet_eval}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
EVAL_LOG="${EVAL_LOG:-${ARTIFACT_DIR}/eval.log}"
RESULT_PREFIX="${RESULT_PREFIX:-${ARTIFACT_DIR}/format_only}"
SUMMARY_PATH="${SUMMARY_PATH:-${ARTIFACT_DIR}/bevdet_eval_summary.json}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "CHECKPOINT_PATH is required" >&2
  exit 2
fi

if [[ -z "${EXTRA_TAG}" ]]; then
  if [[ "${VERSION}" == "v1.0-mini" ]]; then
    EXTRA_TAG="bevdetv3-mini"
  else
    EXTRA_TAG="bevdetv3-nuscenes"
  fi
fi

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

mkdir -p "${ARTIFACT_DIR}"
RESULT_JSON="${RESULT_PREFIX}/pts_bbox/results_nusc.json"
CONTAINER_CHECKPOINT_PATH="$(path_in_container "${CHECKPOINT_PATH}")"
CONTAINER_RESULT_PREFIX="$(path_in_container "${RESULT_PREFIX}")"

CFG_OPTIONS=(
  "data.samples_per_gpu=${SAMPLES_PER_GPU}"
  "data.workers_per_gpu=${WORKERS_PER_GPU}"
  "data.train.dataset.data_root=data/nuscenes/"
  "data.train.dataset.ann_file=data/nuscenes/${EXTRA_TAG}_infos_train.pkl"
  "data.val.data_root=data/nuscenes/"
  "data.val.ann_file=data/nuscenes/${EXTRA_TAG}_infos_val.pkl"
  "data.test.data_root=data/nuscenes/"
  "data.test.ann_file=data/nuscenes/${EXTRA_TAG}_infos_val.pkl"
)

CFG_STRING="${CFG_OPTIONS[*]}"

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
    python tools/test.py ${CONFIG_REL} ${CONTAINER_CHECKPOINT_PATH} --gpu-id 0 --eval mAP \
      --eval-options jsonfile_prefix=${CONTAINER_RESULT_PREFIX} --cfg-options ${CFG_STRING}" \
  2>&1 | tee "${EVAL_LOG}"
eval_status=${PIPESTATUS[0]}
set -e

if [[ ${eval_status} -ne 0 ]]; then
  exit "${eval_status}"
fi

python3 - "${SUMMARY_PATH}" "${CHECKPOINT_PATH}" "${RESULT_JSON}" "${EVAL_LOG}" <<'PY'
from pathlib import Path
import json
import sys

summary_path = Path(sys.argv[1])
payload = {
    "checkpoint_path": str(Path(sys.argv[2])),
    "prediction_path": str(Path(sys.argv[3])),
    "eval_log": str(Path(sys.argv[4])),
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
