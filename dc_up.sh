#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
COMPOSE_PROJECT_DIRECTORY="${VBOGS_COMPOSE_PROJECT_DIRECTORY:-${REPO_ROOT}}"
DEFAULT_COMPOSE_FILES="${REPO_ROOT}/docker/compose/compose.yml:${REPO_ROOT}/docker/compose/dev.yml"
COMPOSE_FILES="${VBOGS_COMPOSE_FILES:-${DEFAULT_COMPOSE_FILES}}"

usage() {
  cat <<'USAGE'
Usage: ./dc_up.sh [SERVICE...]

Start the VBOGS local development Docker Compose stack.

Equivalent to:
  docker compose --project-directory . \
    -f docker/compose/compose.yml \
    -f docker/compose/dev.yml \
    up -d --no-build

Examples:
  ./dc_up.sh
  ./dc_up.sh vbogs-pipeline

Environment:
  VBOGS_COMPOSE_FILES              Colon-separated compose files.
  VBOGS_COMPOSE_PROJECT_DIRECTORY  Compose project directory. Defaults to this repo root.
USAGE
}

case "${1:-}" in
  -h | --help)
    usage
    exit 0
    ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  echo "docker was not found on PATH." >&2
  exit 1
fi

compose_args=(--project-directory "${COMPOSE_PROJECT_DIRECTORY}")
IFS=':' read -r -a compose_files <<< "${COMPOSE_FILES}"
for compose_file in "${compose_files[@]}"; do
  compose_args+=(-f "${compose_file}")
done

exec docker compose "${compose_args[@]}" up -d --no-build "$@"
