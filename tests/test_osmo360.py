import json
from types import SimpleNamespace

import pytest

from vbogs.osmo360 import (
    CROP_SIZE,
    FOV_DEGREES,
    Osmo360ValidationError,
    VideoInfo,
    image_name,
    parse_ffprobe,
    projection_command,
    rig_config,
    sample_timestamps,
    split_timestamp_groups,
    validate_scene_id,
    validate_video_info,
    virtual_cameras,
)
from vbogs.web.progress import project_run_progress
from scripts.run_osmo360_pipeline import run_bundle


def test_balanced_profile_has_six_overlapping_virtual_pinhole_cameras():
    cameras = virtual_cameras()

    assert [camera.name for camera in cameras] == ["front", "right", "back", "left", "up", "down"]
    assert all(camera.width == CROP_SIZE and camera.fov_degrees == FOV_DEGREES for camera in cameras)
    assert cameras[0].focal_length > 0
    assert image_name(12, cameras[0]) == "front/frame_0012.png"


def test_timestamp_sampling_obeys_profile_bounds_and_is_centered():
    short = sample_timestamps(30.0)
    long = sample_timestamps(720.0)

    assert len(short) == 60
    assert len(long) == 180
    assert 0 < short[0] < short[-1] < 30.0
    assert short == sorted(short)


def test_timestamp_group_split_never_splits_virtual_faces():
    splits = split_timestamp_groups(range(10))

    assert sum(len(values) for values in splits.values()) == 10
    assert set().union(*map(set, splits.values())) == set(range(10))


def test_video_validation_requires_stitched_equirectangular_input():
    validate_video_info(VideoInfo(5760, 2880, 60.0, "hevc", 1))
    with pytest.raises(Osmo360ValidationError, match="2:1"):
        validate_video_info(VideoInfo(1920, 1080, 60.0, "h264", 1))
    with pytest.raises(Osmo360ValidationError, match="30 seconds"):
        validate_video_info(VideoInfo(4000, 2000, 12.0, "h264", 1))


def test_ffprobe_and_scene_validation_reject_malformed_uploads():
    with pytest.raises(Osmo360ValidationError, match="exactly one"):
        parse_ffprobe({"streams": []})
    with pytest.raises(Osmo360ValidationError, match="Scene identifier"):
        validate_scene_id("../unsafe")
    assert validate_scene_id("living-room_01") == "living-room_01"


def test_projection_command_and_rig_config_preserve_known_camera_geometry(tmp_path):
    camera = virtual_cameras()[1]
    command = projection_command(
        ffmpeg="ffmpeg", video=tmp_path / "input.mp4", timestamp=1.25,
        camera=camera, output=tmp_path / image_name(0, camera),
    )
    config = rig_config({"front": [1, 2], "right": [3]})

    assert "v360=input=equirect:output=rectilinear" in command[command.index("-vf") + 1]
    assert ":yaw=90" in command[command.index("-vf") + 1]
    assert config[0]["cameras"][0]["ref_sensor"] is True
    assert config[0]["cameras"][1]["cam_from_rig_translation"] == [0.0, 0.0, 0.0]


def test_osmo_workflow_uses_its_own_resume_progress_stages(tmp_path):
    run = {
        "workflow": "osmo360_splat", "start_at": "project", "stop_after": "bundle",
        "status": "running", "workspace_path": str(tmp_path),
    }
    (tmp_path / "pipeline.events.jsonl").write_text(
        '{"type":"run_started","stages":["validate","project","sfm","prepare","train","render","bundle"]}\n'
        '{"type":"stage_started","stage":"project"}\n', encoding="utf-8",
    )

    progress = project_run_progress(run)

    assert progress["current_stage"] == {"name": "project", "index": 1, "total": 6}


def test_bundle_is_portable_rgb_only_and_does_not_include_uploaded_video(tmp_path):
    artifacts = tmp_path / "workspace" / "artifacts"
    model = artifacts / "octree" / "scene" / "run"
    (model / "point_cloud" / "iteration_1").mkdir(parents=True)
    (model / "config.yaml").write_text("model_params:\n  source_path: /private/input\n", encoding="utf-8")
    (artifacts / "train_run.json").write_text(json.dumps({"model_path": str(model)}), encoding="utf-8")
    prepared = artifacts / "colmap" / "scene"
    (prepared / "sparse" / "0").mkdir(parents=True)
    (prepared / "metadata.json").write_text("{}", encoding="utf-8")
    (artifacts / "input_manifest.json").write_text("{}", encoding="utf-8")
    (artifacts / "projections").mkdir()
    (artifacts / "projections" / "projection_manifest.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(output_root=tmp_path / "out", scene_id="scene")

    run_bundle(args, artifacts)

    bundle = args.output_root / "scene"
    assert "../prepared" in (bundle / "model" / "config.yaml").read_text(encoding="utf-8")
    assert (bundle / "osmo360_rgb_splat.zip").is_file()
    assert not (bundle / "input" / "source.mp4").exists()
