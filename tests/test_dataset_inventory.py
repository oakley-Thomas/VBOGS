from vbogs.dataset_inventory import (
    format_clip_table,
    list_kitti360_clips,
    list_nvidia_ncore_clips,
)


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_lists_ready_nvidia_ncore_clip_with_pipeline_args(tmp_path):
    scene_id = "00b769dd-b4fa-4d88-ba4e-e6a230ff0c66"
    scene_dir = tmp_path / "ncore" / scene_id
    touch(scene_dir / f"pai_{scene_id}.json")
    touch(scene_dir / f"pai_{scene_id}.ncore4.zarr.itar")
    touch(scene_dir / f"pai_{scene_id}.ncore4-camera_front_wide_120fov.zarr.itar")
    touch(scene_dir / f"pai_{scene_id}.ncore4-camera_front_tele_30fov.zarr.itar")
    touch(scene_dir / f"pai_{scene_id}.ncore4-lidar_top_360fov.zarr.itar")

    clips = list_nvidia_ncore_clips(tmp_path / "ncore")

    assert len(clips) == 1
    clip = clips[0]
    assert clip.dataset == "nvidia_ncore"
    assert clip.scene_id == scene_id
    assert clip.ready
    assert clip.files["camera_components"] == 2
    assert clip.files["lidar_components"] == 1
    assert clip.pipeline_args[-2:] == ("--scene-id", scene_id)


def test_marks_partial_nvidia_ncore_clip(tmp_path):
    scene_id = "partial_clip"
    scene_dir = tmp_path / "ncore" / scene_id
    touch(scene_dir / f"pai_{scene_id}.ncore4.zarr.itar")

    clips = list_nvidia_ncore_clips(tmp_path / "ncore")

    assert len(clips) == 1
    assert clips[0].status == "partial"
    assert "missing default camera component camera_front_wide_120fov" in clips[0].notes
    assert "missing default lidar component lidar_top_360fov" in clips[0].notes


def test_lists_ready_kitti360_drive(tmp_path):
    drive = "2013_05_28_drive_0004_sync"
    raw_root = tmp_path / "KITTI-360" / "images"
    poses_root = tmp_path / "KITTI-360" / "data_poses"
    calibration_dir = tmp_path / "KITTI-360" / "calibration"
    touch(raw_root / drive / "image_00" / "data_rect" / "0000000000.png")
    touch(raw_root / drive / "image_01" / "data_rect" / "0000000000.png")
    touch(poses_root / drive / "cam0_to_world.txt")
    touch(calibration_dir / "perspective.txt")

    clips = list_kitti360_clips(
        raw_root=raw_root,
        poses_root=poses_root,
        calibration_dir=calibration_dir,
    )

    assert len(clips) == 1
    assert clips[0].dataset == "kitti360"
    assert clips[0].scene_id == drive
    assert clips[0].ready
    assert clips[0].pipeline_args == ("--drive", drive)


def test_format_table_can_include_selectors(tmp_path):
    scene_id = "clip_ready"
    scene_dir = tmp_path / "ncore" / scene_id
    touch(scene_dir / f"pai_{scene_id}.json")
    touch(scene_dir / f"pai_{scene_id}.ncore4.zarr.itar")
    touch(scene_dir / f"pai_{scene_id}.ncore4-camera_front_wide_120fov.zarr.itar")
    touch(scene_dir / f"pai_{scene_id}.ncore4-camera_front_tele_30fov.zarr.itar")
    touch(scene_dir / f"pai_{scene_id}.ncore4-lidar_top_360fov.zarr.itar")

    table = format_clip_table(list_nvidia_ncore_clips(tmp_path / "ncore"), include_commands=True)

    assert "dataset" in table
    assert "Pipeline selectors:" in table
    assert "--dataset-name nvidia_ncore --scene-id clip_ready" in table
