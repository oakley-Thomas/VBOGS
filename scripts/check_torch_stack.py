#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def add_octree_anygs_to_path(repo_root: Path) -> None:
    octree_root = repo_root / "Octree-AnyGS"
    if str(octree_root) not in sys.path:
        sys.path.insert(0, str(octree_root))


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
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    add_octree_anygs_to_path(repo_root)

    torch = importlib.import_module("torch")
    torch_scatter = importlib.import_module("torch_scatter")
    gsplat = importlib.import_module("gsplat")

    from gaussian_renderer.render import render  # noqa: F401

    print(f"repo_root={repo_root}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"torch_scatter={torch_scatter.__version__}")
    print(f"gsplat={getattr(gsplat, '__version__', 'unknown')}")

    cuda_available = torch.cuda.is_available()
    print(f"cuda_available={cuda_available}")
    if not cuda_available:
        raise RuntimeError("CUDA is not available in the current torch environment")

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    print(f"device={device_name}")
    print(f"capability=sm_{capability[0]}{capability[1]}")

    # Exercise a real CUDA op and a torch_scatter CUDA kernel.
    lhs = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    rhs = torch.eye(4, dtype=torch.float32, device=device)
    prod = lhs @ rhs
    if not torch.allclose(prod, lhs):
        raise RuntimeError("Basic CUDA matmul sanity check failed")

    src = torch.tensor(
        [[1.0, 2.0], [3.0, 0.5], [2.5, 4.0], [0.1, 8.0]],
        dtype=torch.float32,
        device=device,
    )
    index = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)
    scattered, argmax = torch_scatter.scatter_max(src, index, dim=0)
    expected = torch.tensor([[2.5, 4.0], [3.0, 8.0]], dtype=torch.float32, device=device)
    if not torch.allclose(scattered, expected):
        raise RuntimeError(
            f"torch_scatter scatter_max sanity check failed: {scattered} != {expected}"
        )

    if argmax.shape != expected.shape:
        raise RuntimeError("torch_scatter returned an unexpected argmax shape")

    print("cuda_matmul=ok")
    print("torch_scatter_cuda=ok")
    print("gaussian_renderer_import=ok")
    print("torch_stack_check=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
