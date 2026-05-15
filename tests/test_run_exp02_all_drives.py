import argparse
import sys
from pathlib import Path

import pytest

from scripts.run_exp02_all_drives import (
    EXP02_CONFIGS,
    build_pipeline_command,
    config_paths,
    discover_drives,
    normalize_extra_args,
)
from vbogs.data_layout import resolve_kitti360_path


def make_drive(raw_root: Path, poses_root: Path, drive: str, *, pose: bool = True) -> None:
    (raw_root / drive / "image_00" / "data_rect").mkdir(parents=True)
    (raw_root / drive / "image_01" / "data_rect").mkdir(parents=True)
    if pose:
        (poses_root / drive).mkdir(parents=True)
        (poses_root / drive / "cam0_to_world.txt").write_text("", encoding="utf-8")


def test_discover_drives_requires_stereo_images_and_pose(tmp_path):
    raw_root = tmp_path / "images"
    poses_root = tmp_path / "poses"
    make_drive(raw_root, poses_root, "2013_05_28_drive_0009_sync")
    make_drive(raw_root, poses_root, "2013_05_28_drive_0007_sync")
    make_drive(raw_root, poses_root, "2013_05_28_drive_0008_sync", pose=False)
    (raw_root / "2013_05_28_drive_0010_sync" / "image_00" / "data_rect").mkdir(
        parents=True
    )

    assert discover_drives(raw_root, poses_root) == [
        "2013_05_28_drive_0007_sync",
        "2013_05_28_drive_0009_sync",
    ]


def test_discover_drives_raises_for_missing_roots(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_drives(tmp_path / "missing-images", tmp_path / "missing-poses")


def test_default_config_is_exp02_implicit_profile():
    args = argparse.Namespace(config=None, variant="implicit")

    assert config_paths(args) == [EXP02_CONFIGS["implicit"]]


def test_both_variant_runs_explicit_then_implicit():
    args = argparse.Namespace(config=None, variant="both")

    assert config_paths(args) == [EXP02_CONFIGS["explicit"], EXP02_CONFIGS["implicit"]]


def test_build_pipeline_command_forwards_roots_and_extra_args():
    args = argparse.Namespace(
        raw_root=Path("/data/KITTI-360/images"),
        poses_root=Path("/data/KITTI-360/data_poses"),
        calibration_dir=Path("/data/KITTI-360/calibration"),
    )

    cmd = build_pipeline_command(
        config=EXP02_CONFIGS["implicit"],
        drive="2013_05_28_drive_0007_sync",
        args=args,
        extra_args=["--use-service-labels", "--skip-up"],
    )

    assert cmd[:6] == [
        sys.executable,
        "scripts/run_drive_pipeline.py",
        "--config",
        "outputs/experiments/exp02_B_implicit3d_baseline_pipeline_config.yaml",
        "--drive",
        "2013_05_28_drive_0007_sync",
    ]
    assert "--raw-root" in cmd
    assert cmd[cmd.index("--poses-root") + 1] == "/data/KITTI-360/data_poses"
    assert cmd[-2:] == ["--use-service-labels", "--skip-up"]


def test_normalize_extra_args_strips_passthrough_separator():
    assert normalize_extra_args(["--", "--use-service-labels"]) == ["--use-service-labels"]


def test_raw_layout_resolver_accepts_kitti360_data_2d_raw(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw_root = tmp_path / "data" / "KITTI-360" / "data_2d_raw"
    raw_root.mkdir(parents=True)

    assert resolve_kitti360_path(None, kind="raw") == Path("data/KITTI-360/data_2d_raw")
