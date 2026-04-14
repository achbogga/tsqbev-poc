#!/usr/bin/env bash
set -euo pipefail

BEVDET_ROOT="${BEVDET_ROOT:-/home/achbogga/projects/BEVDet}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/storage/research/nuscenes}"
BEVDET_IMAGE_TAG="${BEVDET_IMAGE_TAG:-tsqbev-bevdet-official:latest}"
CONFIG_REL="${CONFIG_REL:-configs/bevdet/bevdet-r50-cbgs.py}"
VERSION="${VERSION:-v1.0-mini}"
TSQBEV_ROOT="${TSQBEV_ROOT:-/home/achbogga/projects/tsqbev-poc}"
RESEARCH_ASSETS_ROOT="${RESEARCH_ASSETS_ROOT:-/home/achbogga/projects/research_assets}"
CONTAINER_BEVDET_ROOT="${CONTAINER_BEVDET_ROOT:-/workspace/BEVDet}"
CONTAINER_DATASET_ROOT="${CONTAINER_DATASET_ROOT:-/dataset}"
CONTAINER_TSQBEV_ROOT="${CONTAINER_TSQBEV_ROOT:-/workspace/tsqbev-poc}"
CONTAINER_RESEARCH_ASSETS_ROOT="${CONTAINER_RESEARCH_ASSETS_ROOT:-/workspace/research_assets}"
WORK_DIR="${WORK_DIR:-${TSQBEV_ROOT}/artifacts/public_student_bevdet/work_dir}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"
RESULT_PREFIX="${RESULT_PREFIX:-${WORK_DIR}/probe_format_only}"
TEST_LOG="${TEST_LOG:-${WORK_DIR}/probe_test.log}"
EXTRA_TAG="${EXTRA_TAG:-}"

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

CONTAINER_CHECKPOINT_PATH="$(path_in_container "${CHECKPOINT_PATH}")"
CONTAINER_RESULT_PREFIX="$(path_in_container "${RESULT_PREFIX}")"

mkdir -p "${RESULT_PREFIX}"

if [[ -z "${EXTRA_TAG}" ]]; then
  if [[ "${VERSION}" == "v1.0-mini" ]]; then
    EXTRA_TAG="bevdetv3-mini"
  else
    EXTRA_TAG="bevdetv3-nuscenes"
  fi
fi

if [[ -z "${CFG_STRING:-}" ]]; then
  CFG_OPTIONS=(
    "data.test.data_root=data/nuscenes/"
    "data.test.ann_file=data/nuscenes/${EXTRA_TAG}_infos_val.pkl"
    "data.val.data_root=data/nuscenes/"
    "data.val.ann_file=data/nuscenes/${EXTRA_TAG}_infos_val.pkl"
  )
  CFG_STRING="${CFG_OPTIONS[*]}"
fi

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
