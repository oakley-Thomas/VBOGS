#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def progress(message: str) -> None:
    print(message, flush=True)


def add_octree_anygs_to_path(repo_root: Path) -> None:
    octree_root = repo_root / "Octree-AnyGS"
    if str(octree_root) not in sys.path:
        sys.path.insert(0, str(octree_root))


def format_capability(capability: tuple[int, int]) -> str:
    return f"{capability[0]}.{capability[1]}"


def gsplat_arch_error(device_name: str, capability: tuple[int, int]) -> str:
    arch = format_capability(capability)
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    env_suffix = (
        f" Current TORCH_CUDA_ARCH_LIST={env_arch_list!r}."
        if env_arch_list
        else ""
    )
    return (
        "gsplat CUDA rasterization failed because the installed extension does "
        f"not contain kernels for {device_name} (sm_{capability[0]}{capability[1]})."
        f"{env_suffix}\n"
        "Rebuild the vbogs-torch image with a matching architecture, for example:\n"
        f"  bash scripts/build_stack_serial.sh --torch-cuda-arch-list '{arch}' vbogs-torch\n"
        "For a portable image, leave VBOGS_TORCH_CUDA_ARCH_LIST unset and use the "
        "repo default multi-architecture build."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the vbogs-torch CUDA stack and Octree-AnyGS imports."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the VBOGS repository root.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index used for the runtime kernel checks.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    add_octree_anygs_to_path(repo_root)

    torch = importlib.import_module("torch")
    torch_scatter = importlib.import_module("torch_scatter")
    gsplat = importlib.import_module("gsplat")

    from gaussian_renderer.render import render  # noqa: F401

    progress(f"repo_root={repo_root}")
    progress(f"torch={torch.__version__}")
    progress(f"torch_cuda={torch.version.cuda}")
    progress(f"torch_scatter={torch_scatter.__version__}")
    progress(f"gsplat={getattr(gsplat, '__version__', 'unknown')}")

    cuda_available = torch.cuda.is_available()
    progress(f"cuda_available={cuda_available}")
    if not cuda_available:
        raise RuntimeError("CUDA is not available in the current torch environment")

    device_count = torch.cuda.device_count()
    if args.device_index < 0 or args.device_index >= device_count:
        raise RuntimeError(
            f"CUDA device index {args.device_index} is invalid; "
            f"available CUDA devices: {device_count}"
        )

    device = torch.device(f"cuda:{args.device_index}")
    device_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    progress(f"device={device_name}")
    progress(f"capability=sm_{capability[0]}{capability[1]}")

    # Exercise a real CUDA op and a torch_scatter CUDA kernel.
    progress("cuda_matmul: starting")
    lhs = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    rhs = torch.eye(4, dtype=torch.float32, device=device)
    prod = lhs @ rhs
    torch.cuda.synchronize(device)
    if not torch.allclose(prod, lhs):
        raise RuntimeError("Basic CUDA matmul sanity check failed")
    progress("cuda_matmul=ok")

    progress("torch_scatter_cuda: starting")
    src = torch.tensor(
        [[1.0, 2.0], [3.0, 0.5], [2.5, 4.0], [0.1, 8.0]],
        dtype=torch.float32,
        device=device,
    )
    index = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
    scattered, argmax = torch_scatter.scatter_max(src, index, dim=0)
    torch.cuda.synchronize(device)
    expected = torch.tensor([[2.5, 4.0], [3.0, 8.0]], dtype=torch.float32, device=device)
    if not torch.allclose(scattered, expected):
        raise RuntimeError(
            f"torch_scatter scatter_max sanity check failed: {scattered} != {expected}"
        )

    if argmax.shape != expected.shape:
        raise RuntimeError("torch_scatter returned an unexpected argmax shape")
    progress("torch_scatter_cuda=ok")

    # Exercise gsplat's lazy CUDA extension path, which a plain import does not
    # cover. This mirrors the first renderer call in Octree-AnyGS training.
    progress("gsplat_rasterization: starting")
    means = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32, device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    scales = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32, device=device)
    opacities = torch.tensor([0.9], dtype=torch.float32, device=device)
    colors = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    viewmats = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    Ks = torch.tensor(
        [[[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )
    try:
        render_colors, render_alphas, _info = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=16,
            height=16,
            packed=False,
            render_mode="RGB",
        )
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        if "no kernel image is available for execution on the device" in str(exc):
            print(gsplat_arch_error(device_name, capability), file=sys.stderr)
            return 2
        raise
    if render_colors.shape[:3] != (1, 16, 16) or render_alphas.shape[:3] != (1, 16, 16):
        raise RuntimeError(
            "gsplat rasterization returned unexpected shapes: "
            f"colors={tuple(render_colors.shape)}, alphas={tuple(render_alphas.shape)}"
        )

    progress("gsplat_rasterization=ok")
    progress("gaussian_renderer_import=ok")
    progress("torch_stack_check=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
