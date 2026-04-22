#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_ENV_NAME="${TORCH_ENV_NAME:-vbogs-torch}"
JAX_ENV_NAME="${JAX_ENV_NAME:-vbogs-jax}"
TORCH_PYTHON_VERSION="${TORCH_PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.1}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu128}"
PYG_TORCH_VERSION="${PYG_TORCH_VERSION:-2.7.0}"
GSPLAT_WHEEL_TAG="${GSPLAT_WHEEL_TAG:-pt27cu128}"

TORCH_COMMON_PIP_PACKAGES=(
    plyfile
    tensorboard
    tqdm
    einops
    wandb
    lpips
    laspy
    jaxtyping
    colorama
    opencv-python
    scikit-learn
    matplotlib
    kornia
    pyyaml
    ninja
)

ensure_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is required but was not found on PATH" >&2
        return 1
    fi

    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
}

create_torch_env() {
    ensure_conda

    if conda env list | awk '{print $1}' | grep -qx "${TORCH_ENV_NAME}"; then
        echo "Conda env ${TORCH_ENV_NAME} already exists"
        return 0
    fi

    conda create -y -n "${TORCH_ENV_NAME}" "python=${TORCH_PYTHON_VERSION}" pip
    conda activate "${TORCH_ENV_NAME}"

    pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

    pip install torch_scatter \
        -f "https://data.pyg.org/whl/torch-${PYG_TORCH_VERSION}+${TORCH_CUDA_TAG}.html"

    pip install "${TORCH_COMMON_PIP_PACKAGES[@]}"
    pip install rich gsplat \
        --extra-index-url "https://docs.gsplat.studio/whl/${GSPLAT_WHEEL_TAG}"
}

create_jax_env() {
    ensure_conda

    if conda env list | awk '{print $1}' | grep -qx "${JAX_ENV_NAME}"; then
        echo "Conda env ${JAX_ENV_NAME} already exists"
        return 0
    fi

    conda create -y -n "${JAX_ENV_NAME}" python=3.11 pip
    conda activate "${JAX_ENV_NAME}"
    pip install -e "${ROOT_DIR}/vbgs[gpu]"
}

activate_torch() {
    ensure_conda
    set +u
    conda activate "${TORCH_ENV_NAME}"
    set -u
}

activate_jax() {
    ensure_conda
    set +u
    conda activate "${JAX_ENV_NAME}"
    set -u
}

smoke_test_torch() {
    activate_torch
    (cd "${ROOT_DIR}/Octree-AnyGS" && python render.py --help)
}

verify_torch_gpu() {
    activate_torch
    python -c "import torch; print('torch', torch.__version__); print('torch.version.cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
}

check_torch_stack() {
    activate_torch
    python "${ROOT_DIR}/scripts/check_torch_stack.py" --repo-root "${ROOT_DIR}"
}

smoke_test_jax() {
    activate_jax
    python -c "from vbgs.model.train import fit_gmm_step; print(fit_gmm_step.__name__)"
}

case "${1:-}" in
    create-torch)
        create_torch_env
        ;;
    create-jax)
        create_jax_env
        ;;
    create-all)
        create_torch_env
        create_jax_env
        ;;
    activate-torch)
        activate_torch
        ;;
    activate-jax)
        activate_jax
        ;;
    smoke-test-torch)
        smoke_test_torch
        ;;
    verify-torch-gpu)
        verify_torch_gpu
        ;;
    check-torch-stack)
        check_torch_stack
        ;;
    smoke-test-jax)
        smoke_test_jax
        ;;
    smoke-test-all)
        smoke_test_torch
        smoke_test_jax
        ;;
    *)
        cat <<EOF
Usage: source scripts/envs.sh <command>

Commands:
  create-torch
  create-jax
  create-all
  activate-torch
  activate-jax
  smoke-test-torch
  verify-torch-gpu
  check-torch-stack
  smoke-test-jax
  smoke-test-all

Examples:
  bash scripts/envs.sh create-all
  source scripts/envs.sh activate-torch
  bash scripts/envs.sh verify-torch-gpu
  bash scripts/envs.sh check-torch-stack
  bash scripts/envs.sh smoke-test-jax
EOF
        ;;
esac
