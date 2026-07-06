import json

from vbogs.dataset_inventory import (
    clips_to_json,
    format_clip_table,
    list_dataset_clips,
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
    assert clip.trained is False
    assert clip.latest_stage is None
    assert clip.stage_output_paths == {}
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
    assert clips[0].trained is False
    assert clips[0].latest_stage is None
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
    assert "trained" in table
    assert "latest_stage" in table
    assert "Pipeline selectors:" in table
    assert "--dataset-name nvidia_ncore --scene-id clip_ready" in table


def test_list_dataset_clips_marks_downloaded_clip_pipeline_stage(tmp_path):
    drive = "2013_05_28_drive_0004_sync"
    raw_root = tmp_path / "KITTI-360" / "images"
    poses_root = tmp_path / "KITTI-360" / "data_poses"
    calibration_dir = tmp_path / "KITTI-360" / "calibration"
    colmap_root = tmp_path / "COLMAP"
    octree_root = tmp_path / "OCTREE-ANYGS"
    points_root = tmp_path / "points_world"
    bucket_root = tmp_path / "m4"
    outputs_root = tmp_path / "outputs"
    m6_root = tmp_path / "m6"

    touch(raw_root / drive / "image_00" / "data_rect" / "0000000000.png")
    touch(raw_root / drive / "image_01" / "data_rect" / "0000000000.png")
    touch(poses_root / drive / "cam0_to_world.txt")
    touch(calibration_dir / "perspective.txt")

    touch(colmap_root / drive / "metadata.json")
    touch(octree_root / drive / "config_only" / "config.yaml")
    touch(octree_root / drive / "trained_run" / "config.yaml")
    touch(octree_root / drive / "trained_run" / "point_cloud" / "iteration_7000" / "point_cloud_anchor.ply")
    touch(points_root / drive / "points_world.npz")
    touch(points_root / drive / "points_world_metadata.json")
    touch(bucket_root / drive / "points_norm.npz")
    touch(bucket_root / drive / "pts_by_anchor.npz")
    touch(bucket_root / drive / "norm_params.json")
    touch(bucket_root / drive / "bucket_metadata.json")
    touch(bucket_root / drive / "anchor_posterior.npz")
    touch(bucket_root / drive / "fit_metadata.json")
    touch(bucket_root / drive / "U.npy")
    touch(bucket_root / drive / "uncertainty_components.npz")
    touch(bucket_root / drive / "uncertainty_metadata.json")
    touch(outputs_root / "uncertainty_maps" / drive / "uncertainty_map_metadata.json")
    touch(outputs_root / "uncertainty_maps" / drive / "anchors_uncertainty_all.ply")
    touch(outputs_root / "uncertainty_views" / drive / "metadata.json")

    clips = list_dataset_clips(
        dataset_name="kitti360",
        raw_root=raw_root,
        poses_root=poses_root,
        calibration_dir=calibration_dir,
        colmap_root=colmap_root,
        octree_output_root=octree_root,
        points_root=points_root,
        bucket_root=bucket_root,
        outputs_root=outputs_root,
        m6_root=m6_root,
    )

    assert len(clips) == 1
    clip = clips[0]
    assert clip.ready
    assert clip.trained
    assert clip.latest_stage == "render"
    assert clip.stage_output_paths["train"].endswith("trained_run")
    assert "map-viz" in clip.stage_output_paths


def test_artifact_only_scene_is_merged_into_inventory(tmp_path):
    scene_id = "artifact_scene"
    colmap_root = tmp_path / "COLMAP"
    outputs_root = tmp_path / "outputs"

    (colmap_root / scene_id).mkdir(parents=True)
    (colmap_root / scene_id / "metadata.json").write_text(
        json.dumps({"dataset": "nvidia_ncore", "scene_id": scene_id}),
        encoding="utf-8",
    )
    touch(outputs_root / "v1_0" / scene_id / "run_manifest.json")

    clips = list_dataset_clips(
        dataset_name="all",
        raw_root=tmp_path / "missing-kitti-raw",
        poses_root=tmp_path / "missing-kitti-poses",
        calibration_dir=tmp_path / "missing-kitti-calibration",
        ncore_root=tmp_path / "missing-ncore",
        colmap_root=colmap_root,
        octree_output_root=tmp_path / "missing-octree",
        points_root=tmp_path / "missing-points",
        bucket_root=tmp_path / "missing-bucket",
        outputs_root=outputs_root,
        m6_root=tmp_path / "missing-m6",
    )

    assert len(clips) == 1
    clip = clips[0]
    assert clip.dataset == "nvidia_ncore"
    assert clip.scene_id == scene_id
    assert clip.status == "artifact-only"
    assert not clip.ready
    assert clip.latest_stage == "bundle"
    assert clip.pipeline_args[-2:] == ("--scene-id", scene_id)


def test_config_only_octree_run_does_not_count_as_trained(tmp_path):
    drive = "2013_05_28_drive_0004_sync"
    octree_root = tmp_path / "OCTREE-ANYGS"
    touch(octree_root / drive / "config_only" / "config.yaml")

    clips = list_dataset_clips(
        dataset_name="all",
        raw_root=tmp_path / "missing-kitti-raw",
        poses_root=tmp_path / "missing-kitti-poses",
        calibration_dir=tmp_path / "missing-kitti-calibration",
        ncore_root=tmp_path / "missing-ncore",
        colmap_root=tmp_path / "missing-colmap",
        octree_output_root=octree_root,
        points_root=tmp_path / "missing-points",
        bucket_root=tmp_path / "missing-bucket",
        outputs_root=tmp_path / "missing-outputs",
        m6_root=tmp_path / "missing-m6",
    )

    assert len(clips) == 0


def test_json_includes_stage_outputs(tmp_path):
    drive = "2013_05_28_drive_0004_sync"
    octree_root = tmp_path / "OCTREE-ANYGS"
    touch(octree_root / drive / "trained_run" / "config.yaml")
    touch(octree_root / drive / "trained_run" / "point_cloud" / "iteration_7000" / "point_cloud_anchor.ply")

    clips = list_dataset_clips(
        dataset_name="all",
        raw_root=tmp_path / "missing-kitti-raw",
        poses_root=tmp_path / "missing-kitti-poses",
        calibration_dir=tmp_path / "missing-kitti-calibration",
        ncore_root=tmp_path / "missing-ncore",
        colmap_root=tmp_path / "missing-colmap",
        octree_output_root=octree_root,
        points_root=tmp_path / "missing-points",
        bucket_root=tmp_path / "missing-bucket",
        outputs_root=tmp_path / "missing-outputs",
        m6_root=tmp_path / "missing-m6",
    )

    payload = json.loads(clips_to_json(clips))

    assert payload[0]["trained"] is True
    assert payload[0]["latest_stage"] == "train"
    assert payload[0]["stage_outputs"]["train"].endswith("trained_run")
