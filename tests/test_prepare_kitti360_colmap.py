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
                "R_rect_00: 1 0 0 0 1 0 0 0 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (calibration_dir / "calib_cam_to_velo.txt").write_text(
        "1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8"
    )
    return raw_root, poses_root, calibration_dir


def write_velodyne_scan(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scan = np.zeros((points.shape[0], 4), dtype=np.float32)
    scan[:, :3] = points
    scan.tofile(path)


def make_args(
    tmp_path: Path,
    *,
    training_cameras: str = "left",
    max_frames: int = 1,
    seed_mode: str = "random",
) -> argparse.Namespace:
    raw_root, poses_root, calibration_dir = write_fixture_drive(tmp_path)
    return argparse.Namespace(
        drive="drive_sync",
        raw_root=raw_root,
        poses_root=poses_root,
        calibration_dir=calibration_dir,
        velodyne_root=tmp_path / "velodyne",
        output_root=tmp_path / "COLMAP",
        frame_step=1,
        max_frames=max_frames,
        copy_mode="copy",
        training_cameras=training_cameras,
        seed_mode=seed_mode,
        stereo_max_points=8,
        seed_max_points=None,
        max_points_per_lidar_frame=5000,
        lidar_min_range_m=2.5,
        lidar_max_range_m=80.0,
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


def test_random_seed_mode_records_seed_metadata(tmp_path, monkeypatch):
    patch_ply_writer(monkeypatch)
    dataset_dir = prepare_kitti.prepare_dataset(make_args(tmp_path))

    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed_mode"] == "random"
    assert metadata["seed_metadata"]["seed_source"] == "random"
    assert metadata["seed_metadata"]["seed_point_count"] > 0


def test_parse_cam_to_velo_roundtrip(tmp_path):
    path = tmp_path / "calib_cam_to_velo.txt"
    path.write_text("0 -1 0 0.5 1 0 0 -0.25 0 0 1 2\n", encoding="utf-8")

    transform = prepare_kitti.parse_cam_to_velo(path)

    assert transform.shape == (4, 4)
    assert transform[:3, :4] == pytest.approx(
        np.array([[0, -1, 0, 0.5], [1, 0, 0, -0.25], [0, 0, 1, 2]], dtype=np.float64)
    )
    assert transform[3] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_parse_cam_to_velo_rejects_wrong_length(tmp_path):
    path = tmp_path / "calib_cam_to_velo.txt"
    path.write_text("1 2 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected 12 values"):
        prepare_kitti.parse_cam_to_velo(path)


def test_load_velodyne_scan_reads_xyzi(tmp_path):
    scan_path = tmp_path / "0000000000.bin"
    write_velodyne_scan(scan_path, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    scan = prepare_kitti.load_velodyne_scan(scan_path)

    assert scan.shape == (2, 4)
    assert scan[:, :3] == pytest.approx(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def make_lidar_calibration() -> prepare_kitti.StereoCalibration:
    pinhole = prepare_kitti.PinholeCalibration(width=4, height=4, fx=100, fy=100, cx=2, cy=2)
    return prepare_kitti.StereoCalibration(
        left=pinhole,
        right=pinhole,
        right_center_in_left=np.array([0.5, 0.0, 0.0]),
        baseline_m=0.5,
        r_rect_00=np.eye(3),
    )


def test_lidar_seed_chain_places_points_in_world(tmp_path):
    velodyne_dir = tmp_path / "velodyne_points" / "data"
    write_velodyne_scan(velodyne_dir / "0000000000.bin", np.array([[1.0, 2.0, 3.0]]))

    cam0_to_velo = np.eye(4)
    cam0_to_velo[:3, 3] = [0.5, 0.0, 0.0]
    r_rect = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    velo_to_cam0_rect = r_rect @ np.linalg.inv(cam0_to_velo)

    c2w = np.eye(4)
    c2w[:3, 3] = [10.0, 20.0, 30.0]
    frames = [
        (
            0,
            tmp_path / "missing_left.png",
            tmp_path / "missing_right.png",
            prepare_kitti.FramePose(frame_id=0, c2w=c2w),
        )
    ]

    args = make_args(tmp_path, seed_mode="lidar")
    points, colors, seed_metadata = prepare_kitti.build_sparse_points_from_lidar(
        frames=frames,
        calibration=make_lidar_calibration(),
        velodyne_dir=velodyne_dir,
        velo_to_cam0_rect=velo_to_cam0_rect,
        args=args,
    )

    # p_velo=(1,2,3) -> cam0_unrect=(0.5,2,3) -> rect (rotate z 90deg)=(-2,0.5,3)
    # -> world=(8,20.5,33)
    assert points == pytest.approx(np.array([[8.0, 20.5, 33.0]], dtype=np.float32))
    # Unreadable image means every point keeps the gray fallback color.
    assert colors.tolist() == [[160, 160, 160]]
    assert seed_metadata["seed_source"] == "velodyne"
    assert seed_metadata["seed_frames"] == [0]
    assert seed_metadata["seed_point_count"] == 1
    assert seed_metadata["colored_point_fraction"] == 0.0


def test_lidar_seed_applies_range_filter_and_caps(tmp_path):
    velodyne_dir = tmp_path / "velodyne_points" / "data"
    near_point = [[0.5, 0.0, 0.0]]
    far_point = [[100.0, 0.0, 0.0]]
    kept_points = [[float(i + 3), 0.0, 0.0] for i in range(10)]
    write_velodyne_scan(
        velodyne_dir / "0000000000.bin",
        np.array(near_point + far_point + kept_points),
    )

    frames = [
        (
            0,
            tmp_path / "missing_left.png",
            tmp_path / "missing_right.png",
            prepare_kitti.FramePose(frame_id=0, c2w=np.eye(4)),
        )
    ]
    args = make_args(tmp_path, seed_mode="lidar")
    args.max_points_per_lidar_frame = 4
    args.seed_max_points = 3

    points, colors, seed_metadata = prepare_kitti.build_sparse_points_from_lidar(
        frames=frames,
        calibration=make_lidar_calibration(),
        velodyne_dir=velodyne_dir,
        velo_to_cam0_rect=np.eye(4),
        args=args,
    )

    assert points.shape == (3, 3)
    assert colors.shape == (3, 3)
    assert seed_metadata["seed_point_count"] == 3
    # Range filter must have dropped the near ego return and the far point.
    assert np.all(np.linalg.norm(points, axis=1) >= 2.5)
    assert np.all(np.linalg.norm(points, axis=1) <= 80.0)


def test_lidar_seed_missing_scans_raises_with_escape_hatch(tmp_path):
    velodyne_dir = tmp_path / "velodyne_points" / "data"
    velodyne_dir.mkdir(parents=True)
    frames = [
        (
            0,
            tmp_path / "missing_left.png",
            tmp_path / "missing_right.png",
            prepare_kitti.FramePose(frame_id=0, c2w=np.eye(4)),
        )
    ]

    with pytest.raises(RuntimeError, match="--seed-mode stereo"):
        prepare_kitti.build_sparse_points_from_lidar(
            frames=frames,
            calibration=make_lidar_calibration(),
            velodyne_dir=velodyne_dir,
            velo_to_cam0_rect=np.eye(4),
            args=make_args(tmp_path, seed_mode="lidar"),
        )


def test_prepare_dataset_lidar_mode_writes_seed_metadata(tmp_path, monkeypatch):
    patch_ply_writer(monkeypatch)
    args = make_args(tmp_path, seed_mode="lidar")
    scan_dir = args.velodyne_root / args.drive / "velodyne_points" / "data"
    write_velodyne_scan(scan_dir / "0000000000.bin", np.array([[5.0, 0.0, 1.0]]))

    dataset_dir = prepare_kitti.prepare_dataset(args)

    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed_mode"] == "lidar"
    assert metadata["velodyne_root"] == str(args.velodyne_root)
    assert metadata["seed_metadata"]["seed_source"] == "velodyne"
    assert metadata["seed_metadata"]["seed_point_count"] == 1
    assert metadata["seed_metadata"]["frames_missing_velodyne"] == 0


def test_prepare_dataset_lidar_mode_requires_velodyne_dir(tmp_path, monkeypatch):
    patch_ply_writer(monkeypatch)
    args = make_args(tmp_path, seed_mode="lidar")

    with pytest.raises(FileNotFoundError, match="Velodyne scan directory not found"):
        prepare_kitti.prepare_dataset(args)
