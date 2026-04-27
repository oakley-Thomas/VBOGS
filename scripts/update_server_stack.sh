#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_COMPOSE_BIN="${DOCKER_COMPOSE_BIN:-docker compose}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_REF="${GIT_REF:-main}"
SKIP_PULL=0
NO_CACHE=0

usage() {
    cat <<EOF
Usage: bash scripts/update_server_stack.sh [options]

Update a server-side VBOGS clone, rebuild both Docker images locally, and
recreate the long-lived utility containers while preserving named volumes.

Options:
  --ref <git-ref>       Git branch/tag/ref to pull. Default: ${GIT_REF}
  --remote <name>       Git remote name. Default: ${GIT_REMOTE}
  --skip-pull           Rebuild from the current local checkout without git pull.
  --no-cache            Rebuild images with docker compose build --no-cache.
  -h, --help            Show this help text.

Environment overrides:
  DOCKER_COMPOSE_BIN    Compose command to run, e.g. "docker compose"
  GIT_REMOTE
  GIT_REF

Examples:
  bash scripts/update_server_stack.sh
  bash scripts/update_server_stack.sh --ref feature/live-tuning
  bash scripts/update_server_stack.sh --skip-pull
  bash scripts/update_server_stack.sh --no-cache
EOF
}

log() {
    printf '[server-update] %s\n' "$*"
}

run_compose() {
    # shellcheck disable=SC2086
    ${DOCKER_COMPOSE_BIN} "$@"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)
            GIT_REF="$2"
            shift 2
            ;;
        --remote)
            GIT_REMOTE="$2"
            shift 2
            ;;
        --skip-pull)
            SKIP_PULL=1
            shift
            ;;
        --no-cache)
            NO_CACHE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

cd "${ROOT_DIR}"

if [[ "${SKIP_PULL}" == "0" ]]; then
    log "Fetching ${GIT_REMOTE}/${GIT_REF}"
    git fetch "${GIT_REMOTE}" "${GIT_REF}"

    log "Checking out ${GIT_REF}"
    git checkout "${GIT_REF}"

    log "Pulling latest commits"
    git pull --ff-only "${GIT_REMOTE}" "${GIT_REF}"

    log "Updating submodules"
    git submodule update --init --recursive
else
    log "Skipping git pull and rebuilding the current checkout"
fi

BUILD_ARGS=()
if [[ "${NO_CACHE}" == "1" ]]; then
    BUILD_ARGS+=(--no-cache)
fi

log "Rebuilding docker images"
VBOGS_GIT_REF="$(git rev-parse HEAD)"
export VBOGS_GIT_REF
log "Using VBOGS_GIT_REF=${VBOGS_GIT_REF}"
run_compose build "${BUILD_ARGS[@]}" vbogs-torch vbogs-jax

log "Recreating containers"
run_compose up -d vbogs-torch vbogs-jax

CURRENT_COMMIT="$(git rev-parse --short HEAD)"
log "Update complete at commit ${CURRENT_COMMIT}"
cat <<EOF

Next checks:
  ${DOCKER_COMPOSE_BIN} exec vbogs-torch python scripts/check_torch_stack.py --repo-root /workspace/VBOGS
  ${DOCKER_COMPOSE_BIN} exec vbogs-jax python -c "import jax; print(jax.devices())"
EOF
