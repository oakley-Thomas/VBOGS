#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_pipeline.sh [RUN_DRIVE_PIPELINE_ARGS...]

Run the VBOGS pipeline from inside the vbogs-pipeline container.
The pipeline image must include Docker CLI, and the active compose stack must
mount the host Docker socket into this container.

Example:
  scripts/run_pipeline.sh \
    --drive 2013_05_28_drive_0008_sync \
    --gpu 0 \
    --jax-device 0 \
    --start-at prepare \
    --stop-after bundle
USAGE
}

case "${1:-}" in
  -h | --help)
    usage
    exit 0
    ;;
esac

dry_run=false
for arg in "$@"; do
  if [ "${arg}" = "--dry-run" ]; then
    dry_run=true
  fi
done

if [ "${dry_run}" = "false" ] && ! command -v docker >/dev/null 2>&1; then
  cat >&2 <<'EOF'
docker was not found inside this container.

Rebuild and recreate the local dev pipeline container from outside the container:
  bash scripts/build_stack_serial.sh vbogs-pipeline
  ./dc_up.sh --force-recreate vbogs-pipeline
EOF
  exit 1
fi

if [ "${dry_run}" = "false" ] && ! docker ps >/dev/null 2>&1; then
  cat >&2 <<'EOF'
docker is installed, but this container cannot reach the host Docker daemon.

Recreate vbogs-pipeline with a compose profile that mounts /var/run/docker.sock:
  ./dc_up.sh --force-recreate vbogs-pipeline

If your Docker socket is somewhere else, set DOCKER_HOST_SOCKET before starting
the stack.
EOF
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

config="${VBOGS_PIPELINE_CONFIG:-configs/pipeline/dev.yaml}"
label_project="${VBOGS_COMPOSE_PROJECT:-}"
if [ "${dry_run}" = "false" ] && [ -z "${label_project}" ] && [ -n "${HOSTNAME:-}" ]; then
  label_project="$(
    docker inspect \
      -f '{{ index .Config.Labels "com.docker.compose.project" }}' \
      "${HOSTNAME}" 2>/dev/null || true
  )"
  if [ "${label_project}" = "<no value>" ]; then
    label_project=""
  fi
fi

cmd=(python scripts/run_drive_pipeline.py --config "${config}" --use-service-labels)
if [ -n "${label_project}" ]; then
  cmd+=(--label-project "${label_project}")
fi
cmd+=("$@")

exec "${cmd[@]}"
