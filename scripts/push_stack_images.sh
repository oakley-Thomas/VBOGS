#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/push_stack_images.sh DOCKERHUB_USERNAME VERSION [--tag-only] [service ...]

Tag and push the VBOGS images produced by:
  bash scripts/build_stack_serial.sh

The destination image tags are:
  DOCKERHUB_USERNAME/vbogs-torch:VERSION
  DOCKERHUB_USERNAME/vbogs-jax:VERSION
  DOCKERHUB_USERNAME/vbogs-sfm:VERSION
  DOCKERHUB_USERNAME/vbogs-vbgs-render:VERSION
  DOCKERHUB_USERNAME/vbogs-pipeline:VERSION
  DOCKERHUB_USERNAME/vbogs-web:VERSION

Services:
  vbogs-torch
  vbogs-jax
  vbogs-sfm
  vbogs-vbgs-render
  vbogs-pipeline
  vbogs-web

Options:
  --tag-only    Retag images without pushing.
  -h, --help    Show this help text.

Examples:
  bash scripts/build_stack_serial.sh
  bash scripts/push_stack_images.sh oakleyth v1.0.0
  bash scripts/push_stack_images.sh oakleyth v1.0.0 --tag-only vbogs-torch
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -lt 2 ]; then
  echo "Missing Docker Hub username and version." >&2
  usage >&2
  exit 2
fi

dockerhub_username="$1"
version_name="$2"
shift 2

if [[ ! "${dockerhub_username}" =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
  echo "Docker Hub username/org must be lowercase and may contain letters, digits, dots, underscores, or dashes." >&2
  exit 2
fi

if [[ ! "${version_name}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "Version must be a valid Docker tag: start with a letter, digit, or underscore, then use letters, digits, dots, underscores, or dashes." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DOCKERHUB_NAMESPACE="${dockerhub_username}" \
VBOGS_IMAGE_TAG="${version_name}" \
  bash "${SCRIPT_DIR}/publish_dockerhub_images.sh" "$@"
