from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/experiment05-seed-comparison"


def run_script(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", SCRIPT, *args],
        cwd=REPO_ROOT,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_ncore_requires_scene_id():
    completed = run_script("--dataset", "ncore", "--dry-run", check=False)
    assert completed.returncode == 2
    assert "--scene-id is required" in completed.stderr


def test_rejects_max_frames_not_divisible_by_eight():
    completed = run_script("--max-frames", "15", "--dry-run", check=False)
    assert completed.returncode == 2
    assert "multiple of 8" in completed.stderr


def test_rejects_unknown_dataset():
    completed = run_script("--dataset", "waymo", "--dry-run", check=False)
    assert completed.returncode == 2
    assert "--dataset must be kitti or ncore" in completed.stderr


def test_dry_run_prints_two_phase_commands_per_seed_variant():
    completed = run_script("--dry-run")
    stdout = completed.stdout

    # One arm per seed source, differing only in --seed-mode.
    assert stdout.count("--seed-mode stereo") == 2
    assert stdout.count("--seed-mode lidar") == 2
    # Fairness-relevant flags pinned identically for every variant.
    assert stdout.count("--llffhold 8") == 4
    assert stdout.count("--max-frames 200") == 4
    # Two phases per variant: prepare gate first, then train onward.
    assert stdout.count("--start-at prepare --stop-after prepare") == 2
    assert stdout.count("--start-at train --stop-after bundle") == 2
    assert stdout.count("fairness gate") == 2
    # Per-variant output roots under <dataset>/<scene>/<arm>.
    root = "outputs/experiments/experiment05-seed-comparison/kitti/2013_05_28_drive_0004_sync"
    assert f"--run-output-root {root}/sgbm" in stdout
    assert f"--run-output-root {root}/lidar" in stdout
    # KITTI arms use the drive flag and the KITTI experiment config.
    assert "--drive 2013_05_28_drive_0004_sync" in stdout
    assert "experiment05_kitti_portainer.yaml" in stdout
    # Analysis step is surfaced in dry-run mode.
    assert "analyze_experiment05.py --experiment-root" in stdout


def test_ncore_dry_run_uses_ncore_dataset_flags():
    completed = run_script(
        "--dataset", "ncore", "--scene-id", "pai_fake", "--dry-run"
    )
    stdout = completed.stdout
    assert "--dataset-name nvidia_ncore --scene-id pai_fake" in stdout
    assert "experiment05_ncore_portainer.yaml" in stdout
    root = "outputs/experiments/experiment05-seed-comparison/ncore/pai_fake"
    assert f"--run-output-root {root}/sgbm" in stdout
    assert f"--run-output-root {root}/lidar" in stdout


def test_variant_selection_runs_single_variant():
    completed = run_script("--variant", "lidar", "--dry-run")
    stdout = completed.stdout
    assert "--seed-mode lidar" in stdout
    assert "--seed-mode stereo" not in stdout
    assert stdout.count("--start-at prepare --stop-after prepare") == 1


def test_unknown_variant_rejected():
    completed = run_script("--variant", "colmap", "--dry-run", check=False)
    assert completed.returncode == 2
    assert "--variant must be" in completed.stderr


def test_extra_args_forwarded_before_stage_flags():
    completed = run_script(
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


def test_positional_clip_overrides_default_drive():
    completed = run_script("2013_05_28_drive_0008_sync", "--dry-run")
    stdout = completed.stdout
    assert "--drive 2013_05_28_drive_0008_sync" in stdout
    assert (
        "outputs/experiments/experiment05-seed-comparison/kitti/"
        "2013_05_28_drive_0008_sync/sgbm" in stdout
    )
