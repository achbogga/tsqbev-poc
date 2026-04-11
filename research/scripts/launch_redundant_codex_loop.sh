#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/achbogga/projects/tsqbev-poc"
ARTIFACT_ROOT="${1:-$REPO_ROOT/artifacts/codex_loop_v1}"
DATASET_ROOT="${2:-/home/achbogga/projects/research/nuscenes}"
PRIMARY_SCREEN="${3:-tsqbev_codex_loop_v1}"
WATCHDOG_SCREEN="${4:-tsqbev_codex_loop_v1_watchdog}"

mkdir -p "$ARTIFACT_ROOT"

screen -S "$PRIMARY_SCREEN" -Q select . >/dev/null 2>&1 || \
  screen -dmS "$PRIMARY_SCREEN" bash -lc "\
    cd '$REPO_ROOT' && \
    uv run tsqbev codex-loop \
      --dataset-root '$DATASET_ROOT' \
      --artifact-dir '$ARTIFACT_ROOT' \
      --proposal-path '$REPO_ROOT/docs/paper/tsqbev_frontier_program.md' \
      --teacher-kind cache \
      --teacher-cache-dir '$REPO_ROOT/artifacts/teacher_cache/centerpoint_pointpillar_mini' \
      --max-experiments 3 \
      --sleep-seconds 20 \
      --wait-poll-seconds 15 \
      --no-git-publish \
      > '$ARTIFACT_ROOT/loop.log' 2>&1"

screen -S "$WATCHDOG_SCREEN" -Q select . >/dev/null 2>&1 || \
  screen -dmS "$WATCHDOG_SCREEN" bash -lc "\
    cd '$REPO_ROOT' && \
    exec '$REPO_ROOT/research/scripts/watch_codex_loop.sh' '$ARTIFACT_ROOT' '$DATASET_ROOT' '$PRIMARY_SCREEN'"
