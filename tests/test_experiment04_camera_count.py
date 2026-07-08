from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/experiment04-camera-count"


def run_script(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", SCRIPT, *args],
        cwd=REPO_ROOT,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_requires_scene_id():
    completed = run_script("--dry-run", check=False)
    assert completed.returncode == 2
    assert "--scene-id is required" in completed.stderr


def test_rejects_max_frames_not_divisible_by_eight():
    completed = run_script(
        "--scene-id", "pai_fake", "--max-frames", "15", "--dry-run", check=False
    )
    assert completed.returncode == 2
    assert "multiple of 8" in completed.stderr


def test_dry_run_prints_two_phase_commands_per_variant():
    completed = run_script("--scene-id", "pai_fake", "--dry-run")
    stdout = completed.stdout

    # Both variants, wide camera always first.
    assert "--camera-id camera_front_wide_120fov --frame-step" in stdout
    assert (
        "--camera-id camera_front_wide_120fov "
        "--camera-id camera_front_tele_30fov" in stdout
    )
    # Fairness-relevant flags pinned identically for every variant.
    assert stdout.count("--llffhold 8") == 4
    assert stdout.count("--max-frames 200") == 4
    # Two phases per variant: prepare gate first, then train onward.
    assert stdout.count("--start-at prepare --stop-after prepare") == 2
    assert stdout.count("--start-at train --stop-after bundle") == 2
    assert stdout.count("fairness gate") == 2
    # Per-variant output roots.
    assert "--run-output-root outputs/experiments/experiment04-camera-count/pai_fake/cam1" in stdout
    assert "--run-output-root outputs/experiments/experiment04-camera-count/pai_fake/cam2" in stdout
    # Discovery and analysis steps are surfaced in dry-run mode.
    assert "inspect_nvidia_ncore_clip.py --scene-id pai_fake" in stdout
    assert "analyze_experiment04.py --experiment-root" in stdout


def test_variant_selection_runs_single_variant():
    completed = run_script("--scene-id", "pai_fake", "--variant", "cam2", "--dry-run")
    stdout = completed.stdout
    assert "cam2" in stdout
    assert "--run-output-root outputs/experiments/experiment04-camera-count/pai_fake/cam1" not in stdout
    assert stdout.count("--start-at prepare --stop-after prepare") == 1


def test_unknown_variant_rejected():
    completed = run_script(
        "--scene-id", "pai_fake", "--variant", "cam9", "--dry-run", check=False
    )
    assert completed.returncode == 2
    assert "--variant must be" in completed.stderr


def test_cameras_list_beyond_known_ids_rejected_without_discovery():
    completed = run_script(
        "--scene-id",
        "pai_fake",
        "--cameras-list",
        "1,2,3",
        "--dry-run",
        check=False,
    )
    assert completed.returncode == 2
    assert "camera ids are known" in completed.stderr


def test_extra_args_forwarded_before_stage_flags():
    completed = run_script(
        "--scene-id",
        "pai_fake",
        "--max-frames",
        "16",
        "--dry-run",
        "--",
        "--iterations",
        "2000",
    )
    stdout = completed.stdout
    assert stdout.count("--max-frames 16") == 4
    # Passthrough args must come before the phase stage flags so the
    # two-phase staging always wins.
    assert "--iterations 2000 --start-at prepare --stop-after prepare" in stdout
    assert "--iterations 2000 --start-at train --stop-after bundle" in stdout
