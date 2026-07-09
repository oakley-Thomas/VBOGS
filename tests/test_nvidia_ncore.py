import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.export_points_world import (
    export_lidar_points_from_loader,
    unproject_rectified_to_world,
)
from scripts.prepare_nvidia_ncore_colmap import (
    build_sparse_seed_from_stereo,
    prepare_dataset_from_loader,
)
from vbogs.ncore_adapter import parse_id_list


class FakeModelParameters:
    width = 4
    height = 4
    fx = 1.0
    fy = 1.0
    cx = 1.0
    cy = 1.0


class FakeCameraSensor:
    def __init__(self, sensor_id, images, c2ws, timestamps):
        self.sensor_id = sensor_id
        self.model_parameters = FakeModelParameters()
        self._images = images
        self._c2ws = c2ws
        self._timestamps = timestamps
        self.frames_count = len(images)

    def get_frame_index_range(self):
        return range(self.frames_count)

    def get_frame_timestamp_us(self, frame_index):
        return self._timestamps[frame_index]

    def get_closest_frame_index(self, timestamp_us):
        deltas = [abs(ts - timestamp_us) for ts in self._timestamps]
        return int(np.argmin(deltas))

    def get_frame_image_array(self, frame_index):
        return self._images[frame_index]

    def get_frames_T_sensor_target(self, target, frame_index):
        assert target == "world"
        return self._c2ws[frame_index]


class FakeLidarSensor:
    def __init__(self, points_by_frame, c2ws, timestamps):
        self._points_by_frame = points_by_frame
        self._c2ws = c2ws
        self._timestamps = timestamps
        self.frames_count = len(points_by_frame)

    def get_frame_index_range(self):
        return range(self.frames_count)

    def get_frame_timestamp_us(self, frame_index):
        return self._timestamps[frame_index]

    def get_frame_point_cloud(self, frame_index, **_kwargs):
        return SimpleNamespace(xyz_m_end=self._points_by_frame[frame_index])

    def get_frames_T_sensor_target(self, target, frame_index):
        assert target == "world"
        return self._c2ws[frame_index]


class FakeLoader:
    def __init__(self):
        image_a = np.zeros((4, 4, 3), dtype=np.uint8)
        image_a[1, 1] = [10, 20, 30]
        image_a[1, 2] = [40, 50, 60]
        image_b = np.full((4, 4, 3), 7, dtype=np.uint8)
        c2w = np.eye(4, dtype=np.float64)
        c2w_shift = np.eye(4, dtype=np.float64)
        c2w_shift[:3, 3] = [10.0, 0.0, 0.0]
        self.cameras = {
            "cam_a": FakeCameraSensor("cam_a", [image_a, image_a], [c2w_shift, c2w_shift], [100, 200]),
            "cam_b": FakeCameraSensor("cam_b", [image_b, image_b], [c2w, c2w], [100, 200]),
        }
        lidar_points = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]], dtype=np.float32)
        self.lidars = {
            "lidar_top_360fov": FakeLidarSensor([lidar_points], [c2w_shift], [100])
        }
        self.camera_ids = list(self.cameras)
        self.lidar_ids = list(self.lidars)

    def get_camera_sensor(self, sensor_id):
        return self.cameras[sensor_id]

    def get_lidar_sensor(self, sensor_id):
        return self.lidars[sensor_id]


def test_prepare_ncore_colmap_writes_multi_camera_metadata(tmp_path):
    args = argparse.Namespace(
        scene_id="clip_001",
        ncore_root=tmp_path / "ncore",
        camera_ids=["cam_a,cam_b"],
        lidar_id="lidar_top_360fov",
        output_root=tmp_path / "COLMAP",
        frame_step=1,
        max_frames=2,
        copy_mode="copy",
        seed_mode="random",
        seed_max_points=8,
        max_points_per_lidar_frame=8,
        random_seed=0,
    )

    prepared = prepare_dataset_from_loader(args, FakeLoader())
    dataset_dir = prepared.dataset_dir
    image_names = colmap_image_names(dataset_dir / "sparse" / "0" / "images.txt")
    image_basenames = [name.rsplit("/", maxsplit=1)[-1] for name in image_names]

    assert (dataset_dir / "images" / "cam_a" / "cam_a_0000000000_0000000000.png").exists()
    assert (dataset_dir / "sparse" / "0" / "cameras.txt").read_text().count("PINHOLE") == 2
    assert "cam_b/cam_b_0000000001_0000000001.png" in image_names
    assert len(image_names) == 4
    assert len(set(image_basenames)) == 4
    assert prepared.metadata["dataset"] == "nvidia_ncore"
    assert prepared.metadata["camera_ids"] == ["cam_a", "cam_b"]
    assert prepared.metadata["primary_camera_id"] == "cam_a"
    assert prepared.metadata["num_frames"] == 2
    assert prepared.metadata["num_images"] == 4
    assert set(prepared.metadata["intrinsics"]) == {"cam_a", "cam_b"}
    assert len(prepared.metadata["frame_records"]) == 2
    for record in prepared.metadata["frame_records"]:
        assert record["primary_camera_id"] == "cam_a"
        assert set(record["cameras"]) == {"cam_a", "cam_b"}
        assert record["cameras"]["cam_a"]["image_name"].startswith("cam_a/cam_a_")
        assert record["cameras"]["cam_b"]["image_name"].startswith("cam_b/cam_b_")


def colmap_image_names(path):
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) >= 10:
            names.append(tokens[9])
    return names


def test_ncore_camera_id_parser_preserves_order_and_deduplicates():
    assert parse_id_list(["cam_a,cam_b", "cam_a"], ("default_cam",)) == [
        "cam_a",
        "cam_b",
    ]
    assert parse_id_list(None, ("cam_a", "cam_a", "cam_b")) == ["cam_a", "cam_b"]


def test_lidar_export_transforms_points_and_preserves_contract(tmp_path):
    args = argparse.Namespace(
        dataset_name="nvidia_ncore",
        scene_id="clip_001",
        drive=None,
        output_root=tmp_path / "points_world",
        output_name="points_world.npz",
        write_ply=False,
        lidar_id="lidar_top_360fov",
        camera_ids=["cam_a"],
        frame_step=1,
        max_frames=1,
        max_points_per_frame=0,
        random_seed=0,
    )

    output_path = export_lidar_points_from_loader(args, FakeLoader())
    payload = np.load(output_path)

    assert set(payload.files) == {"xyz", "rgb", "frame_id"}
    np.testing.assert_allclose(
        payload["xyz"],
        np.array([[10.0, 0.0, 2.0], [11.0, 0.0, 2.0]], dtype=np.float32),
    )
    assert payload["rgb"].tolist() == [[10, 20, 30], [40, 50, 60]]
    assert payload["frame_id"].tolist() == [0, 0]


def make_stereo_seed_loader():
    """Two pinhole cameras offset along x viewing a textured plane at 10 m."""

    rng = np.random.default_rng(3)
    height = width = 64
    fx = 50.0
    baseline_m = 1.0
    depth_m = 10.0
    disparity_px = int(fx * baseline_m / depth_m)

    left_image = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    right_image = np.roll(left_image, -disparity_px, axis=1)

    model_parameters = SimpleNamespace(
        width=width, height=height, fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0
    )
    left_c2w = np.eye(4, dtype=np.float64)
    right_c2w = np.eye(4, dtype=np.float64)
    right_c2w[:3, 3] = [baseline_m, 0.0, 0.0]

    loader = FakeLoader()
    left_sensor = FakeCameraSensor("cam_left", [left_image], [left_c2w], [100])
    right_sensor = FakeCameraSensor("cam_right", [right_image], [right_c2w], [100])
    left_sensor.model_parameters = model_parameters
    right_sensor.model_parameters = model_parameters
    loader.cameras = {"cam_left": left_sensor, "cam_right": right_sensor}
    loader.camera_ids = list(loader.cameras)
    return loader, depth_m


def test_stereo_seed_recovers_plane_depth_and_respects_caps():
    pytest.importorskip("cv2")
    loader, depth_m = make_stereo_seed_loader()

    xyz, rgb, seed_metadata = build_sparse_seed_from_stereo(
        loader=loader,
        pair=["cam_left", "cam_right"],
        frame_step=1,
        max_frames=1,
        seed_max_points=50,
        max_points_per_frame=200,
        pixel_step=1,
        num_disparities=16,
        block_size=5,
        min_disparity=2.0,
        max_depth_m=80.0,
        random_seed=0,
    )

    assert xyz.shape[0] > 0
    assert xyz.shape[0] <= 50
    assert rgb.shape == xyz.shape
    assert seed_metadata["seed_stereo_pair"] == ["cam_left", "cam_right"]
    assert seed_metadata["seed_frames"] == [0]
    assert seed_metadata["seed_point_count"] == xyz.shape[0]
    # The synthetic scene is a fronto-parallel plane at depth_m in front of the
    # left camera at the origin, so recovered depths should cluster there.
    median_depth = float(np.median(xyz[:, 2]))
    assert median_depth == pytest.approx(depth_m, rel=0.2)


def test_stereo_seed_rejects_bad_pair():
    pytest.importorskip("cv2")
    loader, _depth_m = make_stereo_seed_loader()

    with pytest.raises(ValueError, match="exactly two camera ids"):
        build_sparse_seed_from_stereo(
            loader=loader,
            pair=["cam_left"],
            frame_step=1,
            max_frames=1,
            seed_max_points=50,
            max_points_per_frame=200,
            pixel_step=1,
            num_disparities=16,
            block_size=5,
            min_disparity=2.0,
            max_depth_m=80.0,
            random_seed=0,
        )


def test_rectified_unprojection_recovers_plane_points():
    disparity = np.array([[10.0]], dtype=np.float32)
    left_rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
    p_left = np.array([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    p_right = np.array([[10.0, 0.0, 0.0, -10.0], [0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    xyz, rgb = unproject_rectified_to_world(
        disparity=disparity,
        left_rect_rgb=left_rgb,
        p_left=p_left,
        p_right=p_right,
        r_rect_left=np.eye(3),
        left_c2w=np.eye(4),
        min_disparity=0.0,
        max_depth_m=10.0,
        pixel_step=1,
        max_points_per_frame=0,
        rng=np.random.default_rng(0),
    )

    np.testing.assert_allclose(xyz, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
    assert rgb.tolist() == [[1, 2, 3]]
