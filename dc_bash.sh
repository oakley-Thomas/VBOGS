#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
COMPOSE_PROJECT_DIRECTORY="${VBOGS_COMPOSE_PROJECT_DIRECTORY:-${REPO_ROOT}}"
DEFAULT_COMPOSE_FILES="${REPO_ROOT}/docker/compose/compose.yml:${REPO_ROOT}/docker/compose/dev.yml"
COMPOSE_FILES="${VBOGS_COMPOSE_FILES:-${VBOGS_COMPOSE_FILE:-${DEFAULT_COMPOSE_FILES}}}"
DEFAULT_SERVICE="vbogs-pipeline"

usage() {
  cat <<'USAGE'
Usage: ./dc_bash.sh [SERVICE]

Open /bin/bash in a running container from the VBOGS Docker Compose stack.
When SERVICE is omitted, uses vbogs-pipeline.

Examples:
  ./dc_bash.sh
  ./dc_bash.sh vbogs-torch
  ./dc_bash.sh vbogs-jax

Environment:
  VBOGS_COMPOSE_FILES              Colon-separated compose files.
  VBOGS_COMPOSE_FILE               Single compose file, used when VBOGS_COMPOSE_FILES is unset.
  VBOGS_COMPOSE_PROJECT_DIRECTORY  Compose project directory. Defaults to this repo root.
USAGE
}

if [ "$#" -gt 1 ]; then
  usage >&2
  exit 2
fi

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

service="${1:-${DEFAULT_SERVICE}}"
services="$(docker compose "${compose_args[@]}" config --services)"
if ! grep -Fxq "${service}" <<< "${services}"; then
  if [ "$#" -eq 0 ]; then
    echo "Default service '${service}' does not exist in this compose stack." >&2
  else
    echo "No service named '${service}' exists in this compose stack." >&2
  fi
  echo "Available services:" >&2
  printf '%s\n' "${services}" >&2
  exit 1
fi

container_id="$(docker compose "${compose_args[@]}" ps -q "${service}" | sed -n '1p')"
if [ -z "${container_id}" ]; then
  stopped_container_id="$(docker compose "${compose_args[@]}" ps -aq "${service}" | sed -n '1p')"
  if [ -n "${stopped_container_id}" ]; then
    state="$(docker inspect -f '{{.State.Status}}' "${stopped_container_id}" 2>/dev/null || true)"
    echo "Service '${service}' has a container, but it is not running${state:+ (${state})}." >&2
  else
    echo "No container exists for service '${service}'." >&2
  fi
  echo "Start the stack first, for example:" >&2
  echo "  ./dc_up.sh" >&2
  exit 1
fi

docker_exec_args=(-i)
if [ -t 0 ] && [ -t 1 ]; then
  docker_exec_args=(-it)
fi

exec docker exec "${docker_exec_args[@]}" "${container_id}" /bin/bash
