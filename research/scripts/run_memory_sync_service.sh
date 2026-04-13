#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/achbogga/projects/tsqbev-poc"
ARTIFACT_ROOT="${1:-$REPO_ROOT/artifacts/codex_loop_v1}"
HARNESS_ROOT="${2:-$ARTIFACT_ROOT/harness_v2}"
ENV_FILE="${TSQBEV_CODEX_LOOP_ENV_FILE:-$HOME/.config/tsqbev/codex-loop.env}"
LOG_PATH="$ARTIFACT_ROOT/memory_sync.log"
LOCK_PATH="$ARTIFACT_ROOT/memory_sync.lock"

mkdir -p "$ARTIFACT_ROOT"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  printf '[memory-sync] %s already_running\n' "$(date --iso-8601=seconds)" >> "$LOG_PATH"
  exit 0
fi

cd "$REPO_ROOT"
{
  printf '[memory-sync] %s start harness_root=%s\n' "$(date --iso-8601=seconds)" "$HARNESS_ROOT"
  uv run tsqbev harness-memory-sync --artifact-dir "$HARNESS_ROOT"
  uv run tsqbev research-report
  printf '[memory-sync] %s done\n' "$(date --iso-8601=seconds)"
} >> "$LOG_PATH" 2>&1
