#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/achbogga/projects/tsqbev-poc"
ARTIFACT_ROOT="${1:-$REPO_ROOT/artifacts/codex_loop_v1}"
DATASET_ROOT="${2:-/home/achbogga/projects/research/nuscenes}"
PRIMARY_SCREEN="${3:-tsqbev_codex_loop_v1}"
STALE_SECONDS="${TSQBEV_CODEX_LOOP_STALE_SECONDS:-900}"
SLEEP_SECONDS="${TSQBEV_CODEX_LOOP_WATCHDOG_SLEEP_SECONDS:-30}"
TEACHER_CACHE_DIR="${TSQBEV_TEACHER_CACHE_DIR:-$REPO_ROOT/artifacts/teacher_cache/centerpoint_pointpillar_mini}"

launch_loop() {
  screen -S "$PRIMARY_SCREEN" -Q select . >/dev/null 2>&1 && screen -S "$PRIMARY_SCREEN" -X quit || true
  mkdir -p "$ARTIFACT_ROOT"
  screen -dmS "$PRIMARY_SCREEN" bash -lc "\
    cd '$REPO_ROOT' && \
    uv run tsqbev codex-loop \
      --dataset-root '$DATASET_ROOT' \
      --artifact-dir '$ARTIFACT_ROOT' \
      --proposal-path '$REPO_ROOT/docs/paper/tsqbev_frontier_program.md' \
      --teacher-kind cache \
      --teacher-cache-dir '$TEACHER_CACHE_DIR' \
      --max-experiments 3 \
      --sleep-seconds 20 \
      --wait-poll-seconds 15 \
      --no-git-publish \
      > '$ARTIFACT_ROOT/loop.log' 2>&1"
}

matching_pids() {
  pgrep -f "tsqbev codex-loop --dataset-root $DATASET_ROOT --artifact-dir $ARTIFACT_ROOT" || true
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
    launch_loop
  elif (( STALE == 1 )); then
    kill $PIDS >/dev/null 2>&1 || true
    sleep 2
    launch_loop
  fi

  sleep "$SLEEP_SECONDS"
done
