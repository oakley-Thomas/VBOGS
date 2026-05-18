import json

import numpy as np
import pytest

from scripts.run_vbgs_kitti_baseline import build_exec_command, parse_args
from scripts.train_vbgs_kitti_baseline import (
    BaselineInput,
    REPO_ROOT,
    load_baseline_input,
    main as train_main,
    resolve_output_root,
    save_norm_params,
)


def write_bucket_fixture(root, drive):
    bucket_root = root / "m4" / drive
    bucket_root.mkdir(parents=True)
    points_norm = np.arange(24, dtype=np.float32).reshape(4, 6)
    np.savez_compressed(
        bucket_root / "points_norm.npz",
        points_norm=points_norm,
        frame_id=np.array([10, 10, 20, 30], dtype=np.int32),
    )
    save_norm_params(
        bucket_root / "norm_params.json",
        {
            "offset": np.arange(6, dtype=np.float32),
            "stdevs": np.ones((6,), dtype=np.float32),
        },
    )
    return bucket_root


def write_stereo_fixture(path):
    path.parent.mkdir(parents=True)
    xyz = np.arange(30, dtype=np.float32).reshape(10, 3)
    rgb = np.arange(30, dtype=np.uint8).reshape(10, 3)
    frame_id = np.arange(10, dtype=np.int32)
    np.savez_compressed(path, xyz=xyz, rgb=rgb, frame_id=frame_id)


def test_auto_prefers_bucket_when_bucket_artifacts_exist(tmp_path):
    drive = "drive_sync"
    bucket_root = write_bucket_fixture(tmp_path, drive)
    stereo_path = tmp_path / "points_world" / drive / "points_world.npz"
    write_stereo_fixture(stereo_path)

    args = type(
        "Args",
        (),
        {
            "drive": drive,
            "input_mode": "auto",
            "bucket_root": bucket_root,
            "points_norm": None,
            "norm_params": None,
            "points_world": stereo_path,
            "max_points": 3,
        },
    )()

    loaded = load_baseline_input(args)

    assert isinstance(loaded, BaselineInput)
    assert loaded.input_mode == "bucket"
    assert loaded.normalization_source == "copied"
    assert loaded.selected_point_count == 4
    assert loaded.frame_count == 3
    assert loaded.point_selection["selection"] == "bucket_exact"


def test_auto_falls_back_to_stereo_and_applies_deterministic_cap(tmp_path):
    drive = "drive_sync"
    stereo_path = tmp_path / "points_world" / drive / "points_world.npz"
    write_stereo_fixture(stereo_path)

    args = type(
        "Args",
        (),
        {
            "drive": drive,
            "input_mode": "auto",
            "bucket_root": tmp_path / "missing_m4",
            "points_norm": None,
            "norm_params": None,
            "points_world": stereo_path,
            "max_points": 4,
        },
    )()

    loaded = load_baseline_input(args)

    assert loaded.input_mode == "stereo"
    assert loaded.normalization_source == "generated"
    assert loaded.source_point_count == 10
    assert loaded.selected_point_count == 4
    assert loaded.frame_count == 4
    assert loaded.point_selection["selection"] == "linspace"
    np.testing.assert_allclose(
        loaded.points_norm[:, 0],
        [-1.3416407, -0.4472136, 0.4472136, 1.3416407],
    )


def test_bucket_mode_requires_bucket_artifacts(tmp_path):
    args = type(
        "Args",
        (),
        {
            "drive": "drive_sync",
            "input_mode": "bucket",
            "bucket_root": tmp_path / "missing",
            "points_norm": None,
            "norm_params": None,
            "points_world": None,
            "max_points": 0,
        },
    )()

    with pytest.raises(FileNotFoundError):
        load_baseline_input(args)


def test_default_output_root_uses_outputs_tree():
    assert resolve_output_root("drive_sync", None) == (
        REPO_ROOT / "outputs" / "vbgs_baseline" / "drive_sync"
    ).resolve()


def test_orchestrator_dry_run_uses_label_resolved_jax_container():
    args = parse_args(
        [
            "--drive",
            "drive_sync",
            "--use-service-labels",
            "--label-project",
            "vbogs",
            "--dry-run",
            "--input-mode",
            "bucket",
            "--n-components",
            "123",
            "--batch-size",
            "7",
            "--no-reassign",
        ]
    )

    cmd = build_exec_command(args)

    assert cmd[:5] == [
        "docker",
        "exec",
        "-i",
        "-w",
        "/workspace/VBOGS",
    ]
    assert cmd[5] == "<vbogs:vbogs-jax-container-by-label>"
    assert cmd[6:9] == ["python", "scripts/train_vbgs_kitti_baseline.py", "--drive"]
    assert "drive_sync" in cmd
    assert "--input-mode" in cmd
    assert cmd[cmd.index("--input-mode") + 1] == "bucket"
    assert "--n-components" in cmd
    assert cmd[cmd.index("--n-components") + 1] == "123"
    assert "--no-reassign" in cmd


def test_tiny_jax_baseline_smoke(tmp_path):
    pytest.importorskip("jax")

    points_path = tmp_path / "points_world.npz"
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rgb = np.array(
        [
            [0, 10, 20],
            [30, 40, 50],
            [60, 70, 80],
            [90, 100, 110],
            [120, 130, 140],
            [150, 160, 170],
        ],
        dtype=np.uint8,
    )
    np.savez_compressed(
        points_path,
        xyz=xyz,
        rgb=rgb,
        frame_id=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
    )

    try:
        train_main(
            [
                "--drive",
                "tiny_drive",
                "--input-mode",
                "stereo",
                "--points-world",
                str(points_path),
                "--output-root",
                str(tmp_path / "out"),
                "--n-components",
                "3",
                "--batch-size",
                "4",
                "--no-reassign",
            ]
        )
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"JAX runtime unavailable for smoke test: {exc}")

    assert (tmp_path / "out" / "model_final.json").exists()
    assert (tmp_path / "out" / "model_final.npz").exists()
    metadata_path = tmp_path / "out" / "baseline_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["n_components"] == 3
    assert metadata["selected_point_count"] == 6
    assert np.isfinite(metadata["mean_elbo_per_point"])
