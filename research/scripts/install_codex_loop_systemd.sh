#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/achbogga/projects/tsqbev-poc"
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
ENV_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/tsqbev"
ENV_FILE="$ENV_DIR/codex-loop.env"
ARTIFACT_ROOT="$REPO_ROOT/artifacts/codex_loop_v1"
DATASET_ROOT="/home/achbogga/projects/research/nuscenes"

mkdir -p "$USER_UNIT_DIR" "$ENV_DIR" "$ARTIFACT_ROOT"

if [[ ! -f "$ENV_FILE" ]]; then
  cat >"$ENV_FILE" <<EOF
TSQBEV_TEACHER_CACHE_DIR=$REPO_ROOT/artifacts/teacher_cache/centerpoint_pointpillar_mini
TSQBEV_CODEX_LOOP_MAX_EXPERIMENTS=3
TSQBEV_CODEX_LOOP_SLEEP_SECONDS=20
TSQBEV_CODEX_LOOP_WAIT_POLL_SECONDS=15
COHERE_API_KEY=${COHERE_API_KEY:-}
TSQBEV_COHERE_API_KEY=${TSQBEV_COHERE_API_KEY:-${COHERE_API_KEY:-}}
EOF
fi

cat >"$USER_UNIT_DIR/tsqbev-codex-loop.service" <<EOF
[Unit]
Description=TSQBEV Codex Loop
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=20

[Service]
Type=simple
WorkingDirectory=$REPO_ROOT
ExecStart=$REPO_ROOT/research/scripts/run_codex_loop_service.sh $ARTIFACT_ROOT $DATASET_ROOT
Restart=always
RestartSec=10
TimeoutStartSec=60
TimeoutStopSec=30
KillMode=mixed

[Install]
WantedBy=default.target
EOF

cat >"$USER_UNIT_DIR/tsqbev-memory-sync.service" <<EOF
[Unit]
Description=TSQBEV Background Memory Sync
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=$REPO_ROOT
ExecStart=$REPO_ROOT/research/scripts/run_memory_sync_service.sh $ARTIFACT_ROOT $ARTIFACT_ROOT/harness_v2
TimeoutStartSec=3h
EOF

cat >"$USER_UNIT_DIR/tsqbev-memory-sync.timer" <<EOF
[Unit]
Description=Periodic TSQBEV Background Memory Sync

[Timer]
OnBootSec=2m
OnUnitActiveSec=45m
Persistent=true

[Install]
WantedBy=timers.target
EOF

cat >"$USER_UNIT_DIR/tsqbev-codex-loop-healthcheck.service" <<EOF
[Unit]
Description=TSQBEV Codex Loop Healthcheck

[Service]
Type=oneshot
ExecStart=$REPO_ROOT/research/scripts/check_codex_loop_health.sh tsqbev-codex-loop.service $ARTIFACT_ROOT 900
EOF

cat >"$USER_UNIT_DIR/tsqbev-codex-loop-healthcheck.timer" <<EOF
[Unit]
Description=Periodic TSQBEV Codex Loop Healthcheck

[Timer]
OnBootSec=2m
OnUnitActiveSec=2m
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now tsqbev-codex-loop.service tsqbev-memory-sync.timer tsqbev-codex-loop-healthcheck.timer

tmp_cron="$(mktemp)"
if crontab -l >"$tmp_cron" 2>/dev/null; then
  python3 - "$tmp_cron" "${tmp_cron}.filtered" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
filtered = [
    line
    for line in source
    if "tsqbev_codex_loop_v1_watchdog" not in line
    and "watch_codex_loop.sh" not in line
]
Path(sys.argv[2]).write_text("\n".join(filtered).rstrip() + "\n", encoding="utf-8")
PY
  crontab "${tmp_cron}.filtered"
  rm -f "${tmp_cron}.filtered"
fi
rm -f "$tmp_cron"
