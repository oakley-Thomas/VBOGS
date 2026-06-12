import argparse
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.prepare_kitti360_colmap as prepare_kitti


def write_fixture_drive(root: Path, drive: str = "drive_sync") -> tuple[Path, Path, Path]:
    raw_root = root / "images"
    poses_root = root / "poses"
    calibration_dir = root / "calibration"
    left_dir = raw_root / drive / "image_00" / "data_rect"
    right_dir = raw_root / drive / "image_01" / "data_rect"
    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)
    poses_dir = poses_root / drive
    poses_dir.mkdir(parents=True)
    calibration_dir.mkdir(parents=True)

    for frame_id in (0, 1):
        name = f"{frame_id:010d}.png"
        (left_dir / name).write_bytes(b"left")
        (right_dir / name).write_bytes(b"right")

    identity = np.eye(4, dtype=np.float64)
    pose_lines = []
    for frame_id in (0, 1):
        pose_lines.append(
            f"{frame_id} " + " ".join(f"{value:.8f}" for value in identity.reshape(-1))
        )
    (poses_dir / "cam0_to_world.txt").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")

    (calibration_dir / "perspective.txt").write_text(
        "\n".join(
            [
                "S_rect_00: 4 4",
                "S_rect_01: 4 4",
                "P_rect_00: 100 0 2 0 0 100 2 0 0 0 1 0",
                "P_rect_01: 100 0 3 -50 0 100 2 0 0 0 1 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return raw_root, poses_root, calibration_dir


def make_args(
    tmp_path: Path,
    *,
    training_cameras: str = "left",
    max_frames: int = 1,
) -> argparse.Namespace:
    raw_root, poses_root, calibration_dir = write_fixture_drive(tmp_path)
    return argparse.Namespace(
        drive="drive_sync",
        raw_root=raw_root,
        poses_root=poses_root,
        calibration_dir=calibration_dir,
        output_root=tmp_path / "COLMAP",
        frame_step=1,
        max_frames=max_frames,
        copy_mode="copy",
        training_cameras=training_cameras,
        seed_mode="random",
        stereo_max_points=8,
        random_seed=0,
    )


def patch_ply_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    def write_tiny_ply(path: Path, _xyz: np.ndarray, _rgb: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="ascii")

    monkeypatch.setattr(prepare_kitti, "write_ply", write_tiny_ply)


def colmap_image_entries(path: Path) -> list[dict]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 10 or not tokens[0].isdigit():
            continue
        entries.append(
            {
                "image_id": int(tokens[0]),
                "tvec": [float(tokens[5]), float(tokens[6]), float(tokens[7])],
                "camera_id": int(tokens[8]),
                "name": tokens[9],
            }
        )
    return entries


def test_prepare_kitti360_default_left_mode_preserves_flat_layout(tmp_path, monkeypatch):
    patch_ply_writer(monkeypatch)
    dataset_dir = prepare_kitti.prepare_dataset(make_args(tmp_path))

    assert (dataset_dir / "images" / "0000000000.png").is_file()
    assert not (dataset_dir / "images" / "image_00").exists()
    assert (dataset_dir / "sparse" / "0" / "cameras.txt").read_text(
        encoding="utf-8"
    ).count("PINHOLE") == 1

    entries = colmap_image_entries(dataset_dir / "sparse" / "0" / "images.txt")
    assert entries == [
        {
            "image_id": 1,
            "tvec": [0.0, 0.0, 0.0],
            "camera_id": 1,
            "name": "0000000000.png",
        }
    ]

    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["training_cameras"] == "left"
    assert metadata["num_frames"] == 1
    assert metadata["num_images"] == 1
    assert metadata["selected_frames"] == [0]
    assert metadata["frame_records"][0]["images"][0]["image_name"] == "0000000000.png"


def test_prepare_kitti360_stereo_mode_writes_two_cameras_and_right_pose(
    tmp_path,
    monkeypatch,
):
    patch_ply_writer(monkeypatch)
    dataset_dir = prepare_kitti.prepare_dataset(
        make_args(tmp_path, training_cameras="stereo", max_frames=2)
    )

    assert (dataset_dir / "images" / "image_00" / "0000000000.png").is_file()
    assert (dataset_dir / "images" / "image_01" / "0000000000.png").is_file()
    cameras_txt = (dataset_dir / "sparse" / "0" / "cameras.txt").read_text(
        encoding="utf-8"
    )
    assert cameras_txt.count("PINHOLE") == 2
    assert "2 PINHOLE 4 4 100.00000000 100.00000000 3.00000000 2.00000000" in cameras_txt

    entries = colmap_image_entries(dataset_dir / "sparse" / "0" / "images.txt")
    assert len(entries) == 4
    assert [entry["name"] for entry in entries[:2]] == [
        "image_00/0000000000.png",
        "image_01/0000000000.png",
    ]
    right_entry = entries[1]
    assert right_entry["camera_id"] == 2
    assert right_entry["tvec"][0] == pytest.approx(-0.5)
    assert right_entry["tvec"][1:] == [0.0, 0.0]

    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["training_cameras"] == "stereo"
    assert metadata["num_frames"] == 2
    assert metadata["num_images"] == 4
    assert metadata["selected_frames"] == [0, 1]
    assert metadata["right_camera_center_in_left"] == pytest.approx([0.5, 0.0, 0.0])
    assert metadata["camera_intrinsics"]["image_01"]["camera_id"] == 2
    assert metadata["frame_records"][0]["images"][1]["image_name"] == (
        "image_01/0000000000.png"
    )
