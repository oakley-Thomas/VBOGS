#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_BIN="${DOCKER_BIN:-docker}"
DOCKERHUB_NAMESPACE="${DOCKERHUB_NAMESPACE:-oakleyth}"
TORCH_IMAGE_NAME="${TORCH_IMAGE_NAME:-vbogs-torch}"
JAX_IMAGE_NAME="${JAX_IMAGE_NAME:-vbogs-jax}"
TIMESTAMP_TAG="${TIMESTAMP_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
PUSH_LATEST="${PUSH_LATEST:-0}"

usage() {
    cat <<EOF
Usage: bash scripts/publish_dockerhub.sh [options]

Builds the VBOGS Docker images, tags them with a UTC timestamp, and pushes
them to Docker Hub.

Options:
  --namespace <name>     Docker Hub namespace/user. Default: ${DOCKERHUB_NAMESPACE}
  --tag <tag>            Explicit tag instead of the generated UTC timestamp.
  --push-latest          Also tag and push :latest for both images.
  --no-push              Build and tag locally without pushing.
  -h, --help             Show this help text.

Environment overrides:
  DOCKER_BIN
  DOCKERHUB_NAMESPACE
  TORCH_IMAGE_NAME
  JAX_IMAGE_NAME
  TIMESTAMP_TAG
  PUSH_LATEST

Examples:
  bash scripts/publish_dockerhub.sh
  bash scripts/publish_dockerhub.sh --namespace mydockeruser
  bash scripts/publish_dockerhub.sh --tag 20260423-173000
  bash scripts/publish_dockerhub.sh --push-latest
EOF
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

log() {
    printf '[publish] %s\n' "$*"
}

PUSH_IMAGES=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace)
            DOCKERHUB_NAMESPACE="$2"
            shift 2
            ;;
        --tag)
            TIMESTAMP_TAG="$2"
            shift 2
            ;;
        --push-latest)
            PUSH_LATEST=1
            shift
            ;;
        --no-push)
            PUSH_IMAGES=0
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

require_command "${DOCKER_BIN}"

TORCH_IMAGE="${DOCKERHUB_NAMESPACE}/${TORCH_IMAGE_NAME}"
JAX_IMAGE="${DOCKERHUB_NAMESPACE}/${JAX_IMAGE_NAME}"
TORCH_TAGGED_IMAGE="${TORCH_IMAGE}:${TIMESTAMP_TAG}"
JAX_TAGGED_IMAGE="${JAX_IMAGE}:${TIMESTAMP_TAG}"

log "Repository root: ${ROOT_DIR}"
log "Torch image: ${TORCH_TAGGED_IMAGE}"
log "JAX image: ${JAX_TAGGED_IMAGE}"

cd "${ROOT_DIR}"

log "Building ${TORCH_TAGGED_IMAGE}"
"${DOCKER_BIN}" build -f docker/torch.Dockerfile -t "${TORCH_TAGGED_IMAGE}" .

log "Building ${JAX_TAGGED_IMAGE}"
"${DOCKER_BIN}" build -f docker/jax.Dockerfile -t "${JAX_TAGGED_IMAGE}" .

if [[ "${PUSH_LATEST}" == "1" ]]; then
    log "Tagging latest aliases"
    "${DOCKER_BIN}" tag "${TORCH_TAGGED_IMAGE}" "${TORCH_IMAGE}:latest"
    "${DOCKER_BIN}" tag "${JAX_TAGGED_IMAGE}" "${JAX_IMAGE}:latest"
fi

if [[ "${PUSH_IMAGES}" == "1" ]]; then
    log "Pushing ${TORCH_TAGGED_IMAGE}"
    "${DOCKER_BIN}" push "${TORCH_TAGGED_IMAGE}"

    log "Pushing ${JAX_TAGGED_IMAGE}"
    "${DOCKER_BIN}" push "${JAX_TAGGED_IMAGE}"

    if [[ "${PUSH_LATEST}" == "1" ]]; then
        log "Pushing ${TORCH_IMAGE}:latest"
        "${DOCKER_BIN}" push "${TORCH_IMAGE}:latest"

        log "Pushing ${JAX_IMAGE}:latest"
        "${DOCKER_BIN}" push "${JAX_IMAGE}:latest"
    fi
else
    log "Skipping push because --no-push was set"
fi

cat <<EOF

Published tag: ${TIMESTAMP_TAG}

Use these in Portainer:
  VBOGS_TORCH_IMAGE=${TORCH_TAGGED_IMAGE}
  VBOGS_JAX_IMAGE=${JAX_TAGGED_IMAGE}
EOF
