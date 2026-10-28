#!/usr/bin/env bash

set -Eeuo pipefail


REPO_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

RUN_SCRIPT="${REPO_ROOT}/scripts/run_scheduled_paper_cycle.sh"

SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

SERVICE_NAME="crypto-volatility-paper.service"
TIMER_NAME="crypto-volatility-paper.timer"

SERVICE_PATH="${SYSTEMD_USER_DIR}/${SERVICE_NAME}"
TIMER_PATH="${SYSTEMD_USER_DIR}/${TIMER_NAME}"


if ! command -v systemctl >/dev/null 2>&1; then
    echo "ERROR: systemctl is not available."
    exit 1
fi


if [[ ! -x "${RUN_SCRIPT}" ]]; then
    echo "ERROR: ${RUN_SCRIPT} is not executable."
    echo
    echo "Run:"
    echo "chmod +x scripts/run_scheduled_paper_cycle.sh"
    exit 1
fi


mkdir -p "${SYSTEMD_USER_DIR}"


cat > "${SERVICE_PATH}" <<EOF
[Unit]
Description=Crypto Volatility Paper-Trading Cycle

[Service]
Type=oneshot

WorkingDirectory=${REPO_ROOT}

ExecStart=/bin/bash ${RUN_SCRIPT}

Environment=PYTHONUNBUFFERED=1

StandardOutput=journal
StandardError=journal
EOF


cat > "${TIMER_PATH}" <<EOF
[Unit]
Description=Daily Crypto Volatility Paper-Trading Timer

[Timer]
OnCalendar=*-*-* 00:15:00 UTC
Persistent=true
AccuracySec=1min

Unit=${SERVICE_NAME}

[Install]
WantedBy=timers.target
EOF


systemctl --user daemon-reload

systemctl --user enable \
    --now \
    "${TIMER_NAME}"


echo
echo "Paper-trading timer installed."
echo
echo "Repository:"
echo "${REPO_ROOT}"
echo
echo "Service:"
echo "${SERVICE_PATH}"
echo
echo "Timer:"
echo "${TIMER_PATH}"
echo
echo "Schedule:"
echo "Daily at 00:15 UTC"
echo
echo "Timer status:"
systemctl --user list-timers \
    "${TIMER_NAME}" \
    --no-pager