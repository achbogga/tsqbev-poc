#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/achbogga/projects/tsqbev-poc"
ARTIFACT_ROOT="${1:-$REPO_ROOT/artifacts/codex_loop_v1}"
DATASET_ROOT="${2:-/home/achbogga/projects/research/nuscenes}"
ENV_FILE="${TSQBEV_CODEX_LOOP_ENV_FILE:-$HOME/.config/tsqbev/codex-loop.env}"
LOG_PATH="$ARTIFACT_ROOT/loop.log"

mkdir -p "$ARTIFACT_ROOT"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

cd "$REPO_ROOT"
exec stdbuf -oL -eL uv run tsqbev codex-loop \
  --dataset-root "$DATASET_ROOT" \
  --artifact-dir "$ARTIFACT_ROOT" \
  --proposal-path "$REPO_ROOT/docs/paper/tsqbev_frontier_program.md" \
  --teacher-kind cache \
  --teacher-cache-dir "${TSQBEV_TEACHER_CACHE_DIR:-$REPO_ROOT/artifacts/teacher_cache/centerpoint_pointpillar_mini}" \
  --max-experiments "${TSQBEV_CODEX_LOOP_MAX_EXPERIMENTS:-3}" \
  --sleep-seconds "${TSQBEV_CODEX_LOOP_SLEEP_SECONDS:-20}" \
  --wait-poll-seconds "${TSQBEV_CODEX_LOOP_WAIT_POLL_SECONDS:-15}" \
  --no-git-publish \
  >> "$LOG_PATH" 2>&1
