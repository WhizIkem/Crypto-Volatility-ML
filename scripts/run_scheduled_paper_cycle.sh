#!/usr/bin/env bash

set -Eeuo pipefail


REPO_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

cd "${REPO_ROOT}"


export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"


timestamp() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}


echo "[$(timestamp)] Starting scheduled paper-trading cycle"


if ! command -v docker >/dev/null 2>&1; then
    echo "[$(timestamp)] ERROR: docker command not found"
    exit 1
fi


if ! docker info >/dev/null 2>&1; then
    echo "[$(timestamp)] ERROR: Docker daemon is unavailable"
    exit 1
fi


echo "[$(timestamp)] Running paper trader"

docker compose run --rm paper-trader


echo "[$(timestamp)] Paper trader completed"

echo "[$(timestamp)] Running reconciliation"

docker compose run --rm reconcile


echo "[$(timestamp)] Reconciliation completed"

echo "[$(timestamp)] Scheduled paper-trading cycle finished successfully"