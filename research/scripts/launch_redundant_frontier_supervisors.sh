#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/achbogga/projects/tsqbev-poc"
ARTIFACT_ROOT="${1:-$REPO_ROOT/artifacts/autoresearch_frontier_v6}"
DATASET_ROOT="${2:-/mnt/storage/research/nuscenes}"
PRIMARY_SCREEN="${3:-tsqbev_frontier_supervisor_v6}"
WATCHDOG_SCREEN="${4:-tsqbev_frontier_supervisor_v6_watchdog}"

screen -S "$PRIMARY_SCREEN" -Q select . >/dev/null 2>&1 || \
  screen -dmS "$PRIMARY_SCREEN" bash -lc "\
    cd '$REPO_ROOT' && \
    export TSQBEV_SUPERVISOR_USE_PROMOTED_HARNESS=1 && \
    export TSQBEV_HARNESS_ROOT='$REPO_ROOT/artifacts/harness_v2' && \
    export TSQBEV_SUPERVISOR_PRE_RUN_BRIEF_TIMEOUT_SECONDS=30 && \
    export TSQBEV_SUPERVISOR_PRE_RUN_SYNC_TIMEOUT_SECONDS=60 && \
    export TSQBEV_SUPERVISOR_BRIEF_TIMEOUT_SECONDS=90 && \
    export TSQBEV_SUPERVISOR_SYNC_TIMEOUT_SECONDS=180 && \
    uv run tsqbev research-supervisor \
      --dataset-root '$DATASET_ROOT' \
      --artifact-dir '$ARTIFACT_ROOT' \
      --max-experiments 4 \
      --sleep-seconds 20 \
      --wait-poll-seconds 20 \
      --teacher-kind cache \
      --teacher-cache-dir '$REPO_ROOT/artifacts/teacher_cache/centerpoint_pointpillar_mini'"

screen -S "$WATCHDOG_SCREEN" -Q select . >/dev/null 2>&1 || \
  screen -dmS "$WATCHDOG_SCREEN" bash -lc "\
    cd '$REPO_ROOT' && \
    exec '$REPO_ROOT/research/scripts/watch_frontier_supervisor.sh' '$ARTIFACT_ROOT' '$DATASET_ROOT' '$PRIMARY_SCREEN'"
