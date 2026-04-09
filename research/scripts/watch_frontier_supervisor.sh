#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/achbogga/projects/tsqbev-poc"
ARTIFACT_ROOT="${1:-$REPO_ROOT/artifacts/autoresearch_frontier_v6}"
DATASET_ROOT="${2:-/mnt/storage/research/nuscenes}"
PRIMARY_SCREEN="${3:-tsqbev_frontier_supervisor_v6}"
STALE_SECONDS="${TSQBEV_SUPERVISOR_STALE_SECONDS:-600}"
SLEEP_SECONDS="${TSQBEV_SUPERVISOR_WATCHDOG_SLEEP_SECONDS:-30}"
TEACHER_CACHE_DIR="${TSQBEV_TEACHER_CACHE_DIR:-$REPO_ROOT/artifacts/teacher_cache/centerpoint_pointpillar_mini}"
HARNESS_ROOT="${TSQBEV_HARNESS_ROOT:-$REPO_ROOT/artifacts/harness_v2}"

launch_supervisor() {
  screen -S "$PRIMARY_SCREEN" -Q select . >/dev/null 2>&1 && screen -S "$PRIMARY_SCREEN" -X quit || true
  screen -dmS "$PRIMARY_SCREEN" bash -lc "\
    cd '$REPO_ROOT' && \
    export TSQBEV_SUPERVISOR_USE_PROMOTED_HARNESS=1 && \
    export TSQBEV_HARNESS_ROOT='$HARNESS_ROOT' && \
    uv run tsqbev research-supervisor \
      --dataset-root '$DATASET_ROOT' \
      --artifact-dir '$ARTIFACT_ROOT' \
      --max-experiments 4 \
      --sleep-seconds 20 \
      --wait-poll-seconds 20 \
      --teacher-kind cache \
      --teacher-cache-dir '$TEACHER_CACHE_DIR'"
}

matching_pids() {
  pgrep -f "tsqbev research-supervisor --dataset-root $DATASET_ROOT --artifact-dir $ARTIFACT_ROOT" || true
}

while true; do
  STATE_PATH="$ARTIFACT_ROOT/state.json"
  PIDS="$(matching_pids | tr '\n' ' ' | xargs echo -n || true)"
  STALE=0
  if [[ -f "$STATE_PATH" ]]; then
    NOW="$(date +%s)"
    MTIME="$(stat -c %Y "$STATE_PATH" 2>/dev/null || echo 0)"
    if (( NOW - MTIME > STALE_SECONDS )); then
      STALE=1
    fi
  fi

  if [[ -z "$PIDS" ]]; then
    launch_supervisor
  elif (( STALE == 1 )); then
    kill $PIDS >/dev/null 2>&1 || true
    sleep 2
    launch_supervisor
  fi

  sleep "$SLEEP_SECONDS"
done
