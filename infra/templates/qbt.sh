#!/bin/bash
# /usr/local/bin/qbt, written by Terraform (infra/templates/qbt.sh): the
# commands for running the service on this instance. Run as root (sudo).
#
#   qbt migrate                         alembic upgrade head, then exits
#   qbt ingest SPY --start 2008-01-01 --end 2025-01-01
#   qbt train  SPY --start 2008-01-01 --end 2025-01-01
#   qbt up                              start (or update) api and worker
#   qbt down                            stop them
#   qbt ps | qbt logs [service]         status, logs
#
# Every command first refreshes DATABASE_URL from SSM Parameter Store into
# /run/qbt/env (tmpfs, root only), so a rotated password is picked up.
set -euo pipefail

source /opt/qbt/qbt.conf   # QBT_REGION, QBT_PARAMETER
compose=(docker compose --project-name qbt --file /opt/qbt/compose.yaml)

refresh_env() {
    install -d -m 0700 /run/qbt
    local url
    url=$(aws ssm get-parameter --region "$QBT_REGION" --name "$QBT_PARAMETER" \
          --with-decryption --query Parameter.Value --output text)
    (umask 077 && printf 'DATABASE_URL=%s\n' "$url" > /run/qbt/env.new)
    mv /run/qbt/env.new /run/qbt/env
}

command=${1:-}
[ $# -gt 0 ] && shift
case "$command" in
    migrate) refresh_env; "${compose[@]}" run --rm migrate ;;
    ingest)  refresh_env; "${compose[@]}" run --rm ingest "$@" ;;
    train)   refresh_env; "${compose[@]}" run --rm train "$@" ;;
    up)      refresh_env; "${compose[@]}" up --detach --wait --wait-timeout 180 api worker ;;
    down)    "${compose[@]}" down ;;
    ps)      "${compose[@]}" ps "$@" ;;
    logs)    "${compose[@]}" logs "$@" ;;
    *) sed -n '2,13p' "$0"; exit 2 ;;
esac
