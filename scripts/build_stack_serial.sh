#!/usr/bin/env bash
set -euo pipefail

# WSL and small local Docker installs can crash when Compose builds the Torch
# and JAX CUDA images at the same time. Build each image explicitly so the
# expensive CUDA wheel download/compile phases do not overlap.

export VBOGS_TORCH_IMAGE="${VBOGS_TORCH_IMAGE:-local/vbogs-torch}"
export VBOGS_JAX_IMAGE="${VBOGS_JAX_IMAGE:-local/vbogs-jax}"
export VBOGS_PIPELINE_IMAGE="${VBOGS_PIPELINE_IMAGE:-local/vbogs-pipeline}"
export COMPOSE_PARALLEL_LIMIT="${COMPOSE_PARALLEL_LIMIT:-1}"
export VBOGS_TORCH_MAX_JOBS="${VBOGS_TORCH_MAX_JOBS:-1}"

DEFAULT_TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0+PTX;12.0+PTX"

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

if [ -z "${VBOGS_TORCH_CUDA_ARCH_LIST:-}" ]; then
  export VBOGS_TORCH_CUDA_ARCH_LIST="${DEFAULT_TORCH_CUDA_ARCH_LIST}"
elif [ "${VBOGS_TORCH_CUDA_ARCH_LIST}" = "auto" ]; then
  detected_arch="$(detect_torch_cuda_arch)"
  export VBOGS_TORCH_CUDA_ARCH_LIST="${detected_arch:-${DEFAULT_TORCH_CUDA_ARCH_LIST}}"
fi

services=("$@")
if [ "${#services[@]}" -eq 0 ]; then
  services=(vbogs-torch vbogs-jax vbogs-pipeline)
fi

for service in "${services[@]}"; do
  echo "Building ${service} with COMPOSE_PARALLEL_LIMIT=${COMPOSE_PARALLEL_LIMIT}"
  if [ "${service}" = "vbogs-torch" ]; then
    echo "Torch CUDA arch list: ${VBOGS_TORCH_CUDA_ARCH_LIST}; build jobs: ${VBOGS_TORCH_MAX_JOBS}"
  fi
  docker compose build "${service}"
done
