#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
)"

cd "$REPO_ROOT"

mkdir -p logs

SCHEDULER_LOG="$REPO_ROOT/logs/paper_scheduler.log"

export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

log_message() {
    local message="$1"
    local timestamp

    timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    printf '[%s] %s\n' "$timestamp" "$message" \
        | tee -a "$SCHEDULER_LOG"
}

if ! command -v docker >/dev/null 2>&1; then
    log_message "ERROR: Docker is not installed or not on PATH"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    log_message "ERROR: Docker daemon is not available"
    exit 1
fi

log_message "Scheduled paper-trading cycle started"

if ! docker compose run --rm paper-trader \
    2>&1 | tee -a "$SCHEDULER_LOG"; then
    log_message "ERROR: Paper-trading cycle failed"
    exit 1
fi

log_message "Paper-trading cycle completed"

if ! docker compose run --rm reconcile \
    2>&1 | tee -a "$SCHEDULER_LOG"; then
    log_message "ERROR: Reconciliation failed"
    exit 1
fi

log_message "Reconciliation completed"
log_message "Scheduled paper-trading cycle finished successfully"