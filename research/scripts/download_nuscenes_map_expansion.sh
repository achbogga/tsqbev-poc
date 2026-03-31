#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/mnt/storage/research/nuscenes}"
ARCHIVE_NAME="${ARCHIVE_NAME:-nuScenes-map-expansion-v1.3.zip}"
MAP_URL="${MAP_URL:-https://zenodo.org/records/15667707/files/nuScenes-map-expansion-v1.3.zip?download=1}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "missing nuScenes dataset root at ${DATASET_ROOT}" >&2
  exit 1
fi

archive_path="${DATASET_ROOT}/${ARCHIVE_NAME}"

if command -v aria2c >/dev/null 2>&1; then
  aria2c \
    --allow-overwrite=true \
    --continue=true \
    --summary-interval=5 \
    --console-log-level=notice \
    -x 8 \
    -s 8 \
    -k 1M \
    -d "${DATASET_ROOT}" \
    -o "${ARCHIVE_NAME}" \
    "${MAP_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -c -O "${archive_path}" "${MAP_URL}"
elif command -v curl >/dev/null 2>&1; then
  curl -L --retry 8 --continue-at - --output "${archive_path}" "${MAP_URL}"
else
  echo "missing downloader: install aria2c, wget, or curl" >&2
  exit 1
fi

unzip -o "${archive_path}" -d "${DATASET_ROOT}/maps"
