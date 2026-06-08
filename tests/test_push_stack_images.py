from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_fake_docker(bin_dir: Path) -> Path:
    docker_log = bin_dir / "docker.log"
    docker = bin_dir / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "${DOCKER_LOG}"
if [ "${1:-}" = "image" ] && [ "${2:-}" = "inspect" ]; then
  exit 0
fi
exit 0
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    return docker_log


def run_push_script(
    tmp_path: Path,
    *args: str,
    check: bool = True,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = write_fake_docker(bin_dir)
    env = {
        **os.environ,
        "DOCKER_LOG": str(docker_log),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    for name in (
        "DOCKERHUB_NAMESPACE",
        "VBOGS_IMAGE_TAG",
        "VBOGS_LOCAL_TORCH_IMAGE",
        "VBOGS_LOCAL_JAX_IMAGE",
        "VBOGS_LOCAL_VBGS_RENDER_IMAGE",
        "VBOGS_LOCAL_PIPELINE_IMAGE",
        "VBOGS_TORCH_PUSH_IMAGE",
        "VBOGS_JAX_PUSH_IMAGE",
        "VBOGS_VBGS_RENDER_PUSH_IMAGE",
        "VBOGS_PIPELINE_PUSH_IMAGE",
    ):
        env.pop(name, None)

    completed = subprocess.run(
        ["bash", "scripts/push_stack_images.sh", *args],
        cwd=REPO_ROOT,
        env=env,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    log_lines = docker_log.read_text(encoding="utf-8").splitlines() if docker_log.exists() else []
    return completed, log_lines


def test_push_script_tags_and_pushes_default_stack(tmp_path: Path):
    completed, log_lines = run_push_script(tmp_path, "acme", "v2.0.0")

    assert "Retag and push complete." in completed.stdout
    assert "tag local/vbogs-torch acme/vbogs-torch:v2.0.0" in log_lines
    assert "tag local/vbogs-jax acme/vbogs-jax:v2.0.0" in log_lines
    assert "tag local/vbogs-vbgs-render acme/vbogs-vbgs-render:v2.0.0" in log_lines
    assert "tag local/vbogs-pipeline acme/vbogs-pipeline:v2.0.0" in log_lines
    assert "push acme/vbogs-torch:v2.0.0" in log_lines
    assert "push acme/vbogs-jax:v2.0.0" in log_lines
    assert "push acme/vbogs-vbgs-render:v2.0.0" in log_lines
    assert "push acme/vbogs-pipeline:v2.0.0" in log_lines


def test_push_script_can_tag_only_one_service(tmp_path: Path):
    completed, log_lines = run_push_script(
        tmp_path,
        "acme",
        "cuda12",
        "--tag-only",
        "vbogs-jax",
    )

    assert "Retag complete." in completed.stdout
    assert log_lines == [
        "image inspect local/vbogs-jax",
        "tag local/vbogs-jax acme/vbogs-jax:cuda12",
    ]


def test_push_script_rejects_invalid_version_before_docker(tmp_path: Path):
    completed, log_lines = run_push_script(
        tmp_path,
        "acme",
        "bad/tag",
        check=False,
    )

    assert completed.returncode == 2
    assert "Version must be a valid Docker tag" in completed.stderr
    assert log_lines == []
