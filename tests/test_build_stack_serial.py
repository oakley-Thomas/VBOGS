from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_fake_docker(bin_dir: Path) -> None:
    docker = bin_dir / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
printf 'DOCKER_ARGS=%s\\n' "$*"
printf 'TORCH_ARCH=%s\\n' "${VBOGS_TORCH_CUDA_ARCH_LIST}"
printf 'RENDER_ARCH=%s\\n' "${VBOGS_RENDER_CUDA_ARCH_LIST}"
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)


def run_build_script(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_fake_docker(bin_dir)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    env.pop("VBOGS_TORCH_CUDA_ARCH_LIST", None)
    env.pop("VBOGS_RENDER_CUDA_ARCH_LIST", None)
    return subprocess.run(
        ["bash", "scripts/build_stack_serial.sh", *args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_cuda_arch_list_sets_torch_and_render_build_args(tmp_path: Path):
    completed = run_build_script(
        tmp_path,
        "--cuda-arch-list",
        "7.5;12.0",
        "vbogs-torch",
        "vbogs-vbgs-render",
    )

    assert "Torch CUDA arch list: 7.5;12.0" in completed.stdout
    assert "VBGS render CUDA arch list: 7.5;12.0" in completed.stdout
    assert completed.stdout.count("TORCH_ARCH=7.5;12.0") == 2
    assert completed.stdout.count("RENDER_ARCH=7.5;12.0") == 2


def test_specific_cuda_arch_lists_override_common_arch_list(tmp_path: Path):
    completed = run_build_script(
        tmp_path,
        "--cuda-arch-list",
        "7.5;12.0",
        "--torch-cuda-arch-list",
        "12.0",
        "--render-cuda-arch-list",
        "7.5",
        "vbogs-torch",
        "vbogs-vbgs-render",
    )

    assert "Torch CUDA arch list: 12.0" in completed.stdout
    assert "VBGS render CUDA arch list: 7.5" in completed.stdout
    assert completed.stdout.count("TORCH_ARCH=12.0") == 2
    assert completed.stdout.count("RENDER_ARCH=7.5") == 2
