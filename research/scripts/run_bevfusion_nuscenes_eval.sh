#!/usr/bin/env bash
set -euo pipefail

BEVFUSION_ROOT="${BEVFUSION_ROOT:-/home/achbogga/projects/bevfusion}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/storage/research/nuscenes}"
IMAGE_TAG="${IMAGE_TAG:-tsqbev-bevfusion-official:latest}"
NUM_GPUS="${NUM_GPUS:-1}"
CONFIG_REL="${CONFIG_REL:-configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-pretrained/bevfusion-det.pth}"
EVAL_KIND="${EVAL_KIND:-bbox}"
TSQBEV_ROOT="${TSQBEV_ROOT:-/home/achbogga/projects/tsqbev-poc}"
CFG_OPTIONS="${CFG_OPTIONS:-data.test.samples_per_gpu=1 data.samples_per_gpu=1}"
TORCH_CACHE_DIR="${TORCH_CACHE_DIR:-/workspace/tsqbev-poc/artifacts/bevfusion_cache/torch_hub}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${TSQBEV_ROOT}/artifacts/bevfusion_repro}"
CHECKPOINT_BASENAME="$(basename "${CHECKPOINT_PATH}")"
CHECKPOINT_STEM="${CHECKPOINT_BASENAME%.pth}"
PATCHED_CHECKPOINT_PATH="${PATCHED_CHECKPOINT_PATH:-pretrained/${CHECKPOINT_STEM}.compat.pth}"

case "${EVAL_KIND}" in
  bbox)
    RUN_LABEL="detection"
    ;;
  map)
    RUN_LABEL="segmentation"
    ;;
  *)
    RUN_LABEL="${EVAL_KIND}"
    ;;
esac

LOG_PATH="${LOG_PATH:-${ARTIFACT_DIR}/bevfusion_${RUN_LABEL}_eval.log}"
SUMMARY_PATH="${SUMMARY_PATH:-${ARTIFACT_DIR}/bevfusion_${EVAL_KIND}_summary.json}"

if [[ ! -d "${BEVFUSION_ROOT}" ]]; then
  echo "missing BEVFusion repo at ${BEVFUSION_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "missing nuScenes dataset root at ${DATASET_ROOT}" >&2
  exit 1
fi

mkdir -p "${ARTIFACT_DIR}"
UPSTREAM_HEAD="$(git -C "${BEVFUSION_ROOT}" rev-parse HEAD)"

set +e
docker run --rm --gpus all --shm-size 16g \
  -v "${BEVFUSION_ROOT}:/workspace/bevfusion" \
  -v "${DATASET_ROOT}:/dataset" \
  -v "${TSQBEV_ROOT}:/workspace/tsqbev-poc" \
  -w /workspace/bevfusion \
  "${IMAGE_TAG}" \
  /bin/bash -lc "mkdir -p data && \
    ln -sfn /dataset data/nuscenes && \
    mkdir -p ${TORCH_CACHE_DIR} && \
    export TORCH_HOME=${TORCH_CACHE_DIR} && \
    export PYTHONPATH=/workspace/tsqbev-poc/compat:/workspace/bevfusion:\$PYTHONPATH && \
    python -m pip install ninja && \
    python /workspace/tsqbev-poc/research/scripts/build_bevfusion_feature_decorator_ext.py \
      --bevfusion-root /workspace/bevfusion && \
    python /workspace/tsqbev-poc/research/scripts/patch_bevfusion_nuscenes_dataset.py \
      --bevfusion-root /workspace/bevfusion && \
    python /workspace/tsqbev-poc/research/scripts/patch_bevfusion_depth_lss.py \
      --bevfusion-root /workspace/bevfusion && \
    python /workspace/tsqbev-poc/research/scripts/patch_bevfusion_checkpoint.py \
      --input /workspace/bevfusion/${CHECKPOINT_PATH} \
      --output /workspace/bevfusion/${PATCHED_CHECKPOINT_PATH} && \
    torchpack dist-run -np ${NUM_GPUS} python tools/test.py \
      ${CONFIG_REL} \
      ${PATCHED_CHECKPOINT_PATH} \
      --eval ${EVAL_KIND} \
      --cfg-options ${CFG_OPTIONS}" 2>&1 | tee "${LOG_PATH}"
docker_status=${PIPESTATUS[0]}
set -e

set +e
python "${TSQBEV_ROOT}/research/scripts/parse_bevfusion_eval_log.py" \
  --log "${LOG_PATH}" \
  --output "${SUMMARY_PATH}" \
  --require-kind "${EVAL_KIND}" \
  --config-rel "${CONFIG_REL}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --dataset-root "${DATASET_ROOT}" \
  --upstream-repo-root "${BEVFUSION_ROOT}" \
  --upstream-head "${UPSTREAM_HEAD}" \
  --docker-exit-code "${docker_status}"
parse_status=$?
set -e

if [[ ${parse_status} -eq 0 ]]; then
  if [[ ${docker_status} -ne 0 ]]; then
    echo "wrapper_note: metrics were emitted successfully; treating docker exit ${docker_status} as post-eval teardown failure" >&2
  fi
  exit 0
fi

if [[ ${docker_status} -ne 0 ]]; then
  exit "${docker_status}"
fi

exit "${parse_status}"
