#!/usr/bin/env bash
set -euo pipefail

# WSL and small local Docker installs can crash when Compose builds the Torch
# and JAX CUDA images at the same time. Build each image explicitly so the
# expensive CUDA wheel download/compile phases do not overlap.

export VBOGS_TORCH_IMAGE="${VBOGS_TORCH_IMAGE:-local/vbogs-torch}"
export VBOGS_JAX_IMAGE="${VBOGS_JAX_IMAGE:-local/vbogs-jax}"
export VBOGS_VBGS_RENDER_IMAGE="${VBOGS_VBGS_RENDER_IMAGE:-local/vbogs-vbgs-render}"
export VBOGS_PIPELINE_IMAGE="${VBOGS_PIPELINE_IMAGE:-local/vbogs-pipeline}"
export VBOGS_WEB_IMAGE="${VBOGS_WEB_IMAGE:-local/vbogs-web}"
export COMPOSE_PARALLEL_LIMIT="${COMPOSE_PARALLEL_LIMIT:-1}"
export VBOGS_TORCH_MAX_JOBS="${VBOGS_TORCH_MAX_JOBS:-1}"
export VBOGS_RENDER_MAX_JOBS="${VBOGS_RENDER_MAX_JOBS:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${VBOGS_COMPOSE_FILE:-${REPO_ROOT}/docker/compose/compose.yml}"
COMPOSE_PROJECT_DIRECTORY="${VBOGS_COMPOSE_PROJECT_DIRECTORY:-${REPO_ROOT}}"

DEFAULT_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0+PTX;12.0+PTX"

usage() {
  cat <<'USAGE'
Usage: bash scripts/build_stack_serial.sh [OPTIONS] [SERVICE...]

Builds VBOGS images one at a time. When no services are listed, builds:
  vbogs-torch vbogs-jax vbogs-vbgs-render vbogs-pipeline vbogs-web

Options:
  --cuda-arch-list ARCHS          CUDA arch list for both Torch and render images.
                                  Use 'auto' to detect GPU 0 with nvidia-smi.
  --torch-cuda-arch-list ARCHS    CUDA arch list for vbogs-torch only.
  --render-cuda-arch-list ARCHS   CUDA arch list for vbogs-vbgs-render only.
  --no-cache, --force             Rebuild without using Docker layer cache.
  -h, --help                      Show this help text.

Examples:
  bash scripts/build_stack_serial.sh --cuda-arch-list '7.5;12.0' vbogs-torch vbogs-vbgs-render
  bash scripts/build_stack_serial.sh --cuda-arch-list auto vbogs-torch
USAGE
}

detect_torch_cuda_arch() {
  detected_arch=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    if detected_arch="$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null)"; then
      detected_arch="${detected_arch%%$'\n'*}"
      detected_arch="${detected_arch//[[:space:]]/}"
    fi
  fi
  printf '%s' "${detected_arch}"
}

resolve_cuda_arch_list() {
  local requested="$1"
  if [ -z "${requested}" ]; then
    printf '%s' "${DEFAULT_CUDA_ARCH_LIST}"
    return
  fi
  if [ "${requested}" = "auto" ]; then
    detected_arch="$(detect_torch_cuda_arch)"
    printf '%s' "${detected_arch:-${DEFAULT_CUDA_ARCH_LIST}}"
    return
  fi
  printf '%s' "${requested}"
}

require_option_value() {
  local option="$1"
  local value="${2:-}"
  if [ -z "${value}" ]; then
    echo "${option} requires a value." >&2
    usage >&2
    exit 2
  fi
}

services=()
build_args=()
cuda_arch_list_arg=""
torch_cuda_arch_list_arg=""
render_cuda_arch_list_arg=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --cuda-arch-list)
      require_option_value "$1" "${2:-}"
      cuda_arch_list_arg="$2"
      shift
      ;;
    --cuda-arch-list=*)
      cuda_arch_list_arg="${1#*=}"
      require_option_value "--cuda-arch-list" "${cuda_arch_list_arg}"
      ;;
    --torch-cuda-arch-list)
      require_option_value "$1" "${2:-}"
      torch_cuda_arch_list_arg="$2"
      shift
      ;;
    --torch-cuda-arch-list=*)
      torch_cuda_arch_list_arg="${1#*=}"
      require_option_value "--torch-cuda-arch-list" "${torch_cuda_arch_list_arg}"
      ;;
    --render-cuda-arch-list)
      require_option_value "$1" "${2:-}"
      render_cuda_arch_list_arg="$2"
      shift
      ;;
    --render-cuda-arch-list=*)
      render_cuda_arch_list_arg="${1#*=}"
      require_option_value "--render-cuda-arch-list" "${render_cuda_arch_list_arg}"
      ;;
    --no-cache | --force)
      build_args+=(--no-cache)
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      services+=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      services+=("$1")
      ;;
  esac
  shift
done

if [ -n "${cuda_arch_list_arg}" ]; then
  export VBOGS_TORCH_CUDA_ARCH_LIST="${cuda_arch_list_arg}"
  export VBOGS_RENDER_CUDA_ARCH_LIST="${cuda_arch_list_arg}"
fi
if [ -n "${torch_cuda_arch_list_arg}" ]; then
  export VBOGS_TORCH_CUDA_ARCH_LIST="${torch_cuda_arch_list_arg}"
fi
if [ -n "${render_cuda_arch_list_arg}" ]; then
  export VBOGS_RENDER_CUDA_ARCH_LIST="${render_cuda_arch_list_arg}"
fi

export VBOGS_TORCH_CUDA_ARCH_LIST="$(
  resolve_cuda_arch_list "${VBOGS_TORCH_CUDA_ARCH_LIST:-}"
)"
export VBOGS_RENDER_CUDA_ARCH_LIST="$(
  resolve_cuda_arch_list "${VBOGS_RENDER_CUDA_ARCH_LIST:-}"
)"

if [ "${#services[@]}" -eq 0 ]; then
  services=(vbogs-torch vbogs-jax vbogs-vbgs-render vbogs-pipeline vbogs-web)
fi

for service in "${services[@]}"; do
  echo "Building ${service} with COMPOSE_PARALLEL_LIMIT=${COMPOSE_PARALLEL_LIMIT}"
  if [ "${#build_args[@]}" -gt 0 ]; then
    echo "Docker compose build args: ${build_args[*]}"
  fi
  if [ "${service}" = "vbogs-torch" ]; then
    echo "Torch CUDA arch list: ${VBOGS_TORCH_CUDA_ARCH_LIST}; build jobs: ${VBOGS_TORCH_MAX_JOBS}"
  elif [ "${service}" = "vbogs-vbgs-render" ]; then
    echo "VBGS render CUDA arch list: ${VBOGS_RENDER_CUDA_ARCH_LIST:-default}; build jobs: ${VBOGS_RENDER_MAX_JOBS}"
  fi
  docker compose --project-directory "${COMPOSE_PROJECT_DIRECTORY}" -f "${COMPOSE_FILE}" build "${build_args[@]}" "${service}"
done
