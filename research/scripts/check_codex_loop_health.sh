#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${1:-tsqbev-codex-loop.service}"
ARTIFACT_ROOT="${2:-/home/achbogga/projects/tsqbev-poc/artifacts/codex_loop_v1}"
STALE_SECONDS="${3:-900}"
LOG_PATH="$ARTIFACT_ROOT/daemon_health.log"
HEARTBEAT_PATH="$ARTIFACT_ROOT/heartbeat.json"
SUPERVISOR_ROOT="$ARTIFACT_ROOT/supervisor"

mkdir -p "$ARTIFACT_ROOT"

now="$(date +%s)"
status="ok"
reason="healthy"

if ! systemctl --user is-active --quiet "$SERVICE_NAME"; then
  systemctl --user restart "$SERVICE_NAME" >/dev/null 2>&1 || true
  status="restart"
  reason="service_inactive"
elif [[ ! -f "$HEARTBEAT_PATH" ]]; then
  systemctl --user restart "$SERVICE_NAME" >/dev/null 2>&1 || true
  status="restart"
  reason="missing_heartbeat"
else
  mtime="$(stat -c %Y "$HEARTBEAT_PATH" 2>/dev/null || echo 0)"
  if (( now - mtime > STALE_SECONDS )); then
    systemctl --user restart "$SERVICE_NAME" >/dev/null 2>&1 || true
    status="restart"
    reason="stale_heartbeat"
  else
    read -r active_phase phase_started_at supervisor_phase supervisor_generated_at supervisor_run_activity < <(
      python3 - "$HEARTBEAT_PATH" "$SUPERVISOR_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


heartbeat = load_json(Path(sys.argv[1]))
supervisor_root = Path(sys.argv[2])
supervisor_state = load_json(supervisor_root / "state.json")
latest_activity = 0
if supervisor_root.exists():
    for path in supervisor_root.rglob("*"):
        if path.is_file():
            try:
                latest_activity = max(latest_activity, int(path.stat().st_mtime))
            except OSError:
                pass
print(
    heartbeat.get("active_phase", ""),
    heartbeat.get("active_phase_started_at_utc", ""),
    supervisor_state.get("active_phase", ""),
    supervisor_state.get("generated_at_utc", ""),
    latest_activity,
)
PY
    )
    phase_started_epoch="$(python3 - "$phase_started_at" <<'PY'
from __future__ import annotations
from datetime import datetime
import sys

raw = sys.argv[1].strip()
if not raw:
    print(0)
    raise SystemExit(0)
try:
    print(int(datetime.fromisoformat(raw).timestamp()))
except Exception:
    print(0)
PY
    )"
    supervisor_generated_epoch="$(python3 - "$supervisor_generated_at" <<'PY'
from __future__ import annotations
from datetime import datetime
import sys

raw = sys.argv[1].strip()
if not raw:
    print(0)
    raise SystemExit(0)
try:
    print(int(datetime.fromisoformat(raw).timestamp()))
except Exception:
    print(0)
PY
    )"
    phase_age=0
    if (( phase_started_epoch > 0 )); then
      phase_age=$(( now - phase_started_epoch ))
    fi
    supervisor_age=0
    if (( supervisor_generated_epoch > 0 )); then
      supervisor_age=$(( now - supervisor_generated_epoch ))
    fi
    run_activity_age=0
    if [[ "$supervisor_run_activity" =~ ^[0-9]+$ ]] && (( supervisor_run_activity > 0 )); then
      run_activity_age=$(( now - supervisor_run_activity ))
    fi

    case "$active_phase" in
      harness_search|harness_promote|memory_refresh_request)
        if (( phase_age > STALE_SECONDS )); then
          systemctl --user restart "$SERVICE_NAME" >/dev/null 2>&1 || true
          status="restart"
          reason="stuck_phase_${active_phase}"
        fi
        ;;
      supervisor)
        if [[ ! -f "$SUPERVISOR_ROOT/state.json" ]] && (( phase_age > STALE_SECONDS )); then
          systemctl --user restart "$SERVICE_NAME" >/dev/null 2>&1 || true
          status="restart"
          reason="missing_supervisor_state"
        elif [[ "$supervisor_phase" == "launching_bounded_loop" ]] && (( supervisor_age > STALE_SECONDS )) && (( run_activity_age > STALE_SECONDS )); then
          systemctl --user restart "$SERVICE_NAME" >/dev/null 2>&1 || true
          status="restart"
          reason="stuck_launching_bounded_loop"
        elif (( supervisor_age > STALE_SECONDS * 4 )) && (( run_activity_age > STALE_SECONDS * 4 )); then
          systemctl --user restart "$SERVICE_NAME" >/dev/null 2>&1 || true
          status="restart"
          reason="stale_supervisor_activity"
        fi
        ;;
    esac
  fi
fi

printf '[codex-health] %s service=%s status=%s reason=%s\n' \
  "$(date --iso-8601=seconds)" "$SERVICE_NAME" "$status" "$reason" >> "$LOG_PATH"
