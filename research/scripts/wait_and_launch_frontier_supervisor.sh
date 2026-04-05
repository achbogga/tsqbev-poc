#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/achbogga/projects/tsqbev-poc"
DATASET_ROOT="/home/achbogga/projects/research/nuscenes"
ARTIFACT_ROOT="${1:-$REPO_ROOT/artifacts/autoresearch_frontier}"
TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-$REPO_ROOT/artifacts/teacher_cache/centerpoint_pointpillar_mini}"
WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-}"
POLL_SECONDS="${POLL_SECONDS:-30}"

if [[ -n "$WAIT_FOR_SESSION" ]]; then
  while screen -ls | grep -q "${WAIT_FOR_SESSION}"; do
    sleep "$POLL_SECONDS"
  done
fi

cd "$REPO_ROOT"
rm -f "$ARTIFACT_ROOT/STOP"
export TSQBEV_SUPERVISOR_PRE_RUN_SYNC="${TSQBEV_SUPERVISOR_PRE_RUN_SYNC:-0}"
export TSQBEV_SUPERVISOR_RUN_ON_REJECT="${TSQBEV_SUPERVISOR_RUN_ON_REJECT:-1}"
uv run tsqbev research-supervisor \
  --dataset-root "$DATASET_ROOT" \
  --artifact-dir "$ARTIFACT_ROOT" \
  --teacher-cache-dir "$TEACHER_CACHE_DIR" \
  --max-experiments 4 \
  --sleep-seconds 30 \
  --wait-poll-seconds 20 \
  --git-publish
