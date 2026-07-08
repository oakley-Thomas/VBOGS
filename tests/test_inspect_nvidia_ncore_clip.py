from scripts.inspect_nvidia_ncore_clip import (
    camera_flag_sets,
    inspect_clip_from_loader,
    ordered_camera_ids,
    suggest_max_frames,
)
from tests.test_nvidia_ncore import FakeLoader


def test_inspect_clip_reports_cameras_and_lidars():
    info = inspect_clip_from_loader(FakeLoader())

    assert info["camera_ids"] == ["cam_a", "cam_b"]
    assert info["cameras"] == {
        "cam_a": {"frame_count": 2},
        "cam_b": {"frame_count": 2},
    }
    assert info["lidar_ids"] == ["lidar_top_360fov"]


def test_suggest_max_frames_rounds_down_to_multiple_of_eight():
    assert suggest_max_frames(400, 2) == 200
    assert suggest_max_frames(100, 2) == 48
    assert suggest_max_frames(15, 1) == 8
    assert suggest_max_frames(7, 1) == 0
    assert suggest_max_frames(17, 2) == 8


def test_ordered_camera_ids_prefers_known_defaults():
    ordered = ordered_camera_ids(
        [
            "camera_rear_tele_30fov",
            "camera_front_wide_120fov",
            "camera_cross_left_120fov",
        ]
    )
    assert ordered == [
        "camera_front_wide_120fov",
        "camera_cross_left_120fov",
        "camera_rear_tele_30fov",
    ]


def test_camera_flag_sets_grow_by_one_camera():
    flag_sets = camera_flag_sets(["cam_b", "cam_a"])
    assert flag_sets == [
        "--camera-id cam_a",
        "--camera-id cam_a --camera-id cam_b",
    ]
