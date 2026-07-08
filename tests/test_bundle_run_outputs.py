import json
import zipfile
from pathlib import Path

import numpy as np

from scripts.bundle_run_outputs import bundle_run_outputs


def write_text(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_bundle_run_outputs_copies_curated_artifacts_and_manifest(tmp_path):
    drive = "2013_05_28_drive_0007_sync"
    run_output_dir = tmp_path / "outputs" / "v1_0" / drive

    points_dir = tmp_path / "points_world" / drive
    points_dir.mkdir(parents=True)
    np.savez_compressed(points_dir / "points_world.npz", xyz=np.zeros((2, 3), dtype=np.float32))
    (points_dir / "points_world.ply").write_bytes(b"ply\n")
    write_json(
        points_dir / "points_world_metadata.json",
        {"num_frames": 1000, "num_points": 2, "matcher": "sgbm"},
    )

    bucket_root = tmp_path / "m4" / drive
    bucket_root.mkdir(parents=True)
    np.save(bucket_root / "U.npy", np.array([1.0, 2.0], dtype=np.float32))
    np.savez_compressed(bucket_root / "uncertainty_components.npz", weights=np.ones((1, 1)))
    write_json(bucket_root / "uncertainty_metadata.json", {"anchor_count": 2})
    (bucket_root / "uncertainty_histogram.png").write_bytes(b"png")

    colmap_root = tmp_path / "COLMAP"
    selected_frames = list(range(1000))
    write_json(
        colmap_root / drive / "metadata.json",
        {"num_frames": 1000, "selected_frames": selected_frames},
    )

    model_path = tmp_path / "OCTREE-ANYGS" / drive / "2026-05-08_00:00:00"
    model_path.mkdir(parents=True)
    (model_path / "config.yaml").write_text("model: test\n", encoding="utf-8")
    write_json(model_path / "results.json", {"ours_30000": {"PSNR": 25.0}})
    write_json(model_path / "per_view.json", {"ours_30000": {"PSNR": {"00000.png": 25.0}}})
    write_text(run_output_dir / "views" / "train" / "side_by_side" / "000001.png", "png")

    manifest = bundle_run_outputs(
        drive=drive,
        run_output_dir=run_output_dir,
        points_root=tmp_path / "points_world",
        bucket_root=bucket_root,
        colmap_root=colmap_root,
        octree_output_root=tmp_path / "OCTREE-ANYGS",
        model_path=None,
        map_viz_output_dir=run_output_dir / "pointclouds" / "anchors",
        render_output_dir=run_output_dir / "views",
        nbv_output_dir=run_output_dir / "nbv",
        viewer_export_iteration=-1,
        viewer_export_output_dir=None,
        viewer_export_archive_path=None,
        skip_local_viewer_export=True,
    )

    assert (run_output_dir / "pointclouds" / "stereo" / "points_world.npz").exists()
    assert (run_output_dir / "pointclouds" / "stereo" / "points_world.ply").exists()
    assert (run_output_dir / "uncertainty" / "U.npy").exists()
    assert (run_output_dir / "prepared" / "metadata.json").exists()
    assert (run_output_dir / "octree" / "config.yaml").exists()
    assert (run_output_dir / "octree" / "results.json").exists()
    assert (run_output_dir / "octree" / "per_view.json").exists()
    assert (run_output_dir / "run_manifest.json").exists()
    archive_path = run_output_dir / f"{drive}.zip"
    assert archive_path.exists()

    saved_manifest = json.loads((run_output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["drive"] == drive
    assert saved_manifest["archive"]["path"] == str(archive_path.resolve())
    assert saved_manifest["archive"]["excluded_top_level_dirs"] == ["views"]
    assert saved_manifest["frame_counts"]["num_frames"] == 1000
    assert saved_manifest["frame_counts"]["selected_frame_count"] == 1000
    assert saved_manifest["stereo"]["num_points"] == 2
    assert saved_manifest["stage_outputs"]["rendered_views"] == str((run_output_dir / "views").resolve())
    assert saved_manifest["source_paths"]["octree_model_path"] == str(model_path.resolve())
    assert manifest["missing_optional_artifacts"] == []
    assert manifest["archive"]["path"] == str(archive_path.resolve())

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert f"{drive}/run_manifest.json" in names
    assert f"{drive}/pointclouds/stereo/points_world.npz" in names
    assert f"{drive}/views/train/side_by_side/000001.png" not in names


def test_bundle_run_outputs_records_optional_missing_ply(tmp_path):
    drive = "drive_sync"
    run_output_dir = tmp_path / "out" / drive

    points_dir = tmp_path / "points_world" / drive
    points_dir.mkdir(parents=True)
    np.savez_compressed(points_dir / "points_world.npz", xyz=np.zeros((1, 3), dtype=np.float32))
    write_json(points_dir / "points_world_metadata.json", {"num_frames": 1, "num_points": 1})

    bucket_root = tmp_path / "m4" / drive
    bucket_root.mkdir(parents=True)
    np.save(bucket_root / "U.npy", np.array([1.0], dtype=np.float32))
    np.savez_compressed(bucket_root / "uncertainty_components.npz", weights=np.ones((1, 1)))
    write_json(bucket_root / "uncertainty_metadata.json", {"anchor_count": 1})

    colmap_root = tmp_path / "COLMAP"
    write_json(colmap_root / drive / "metadata.json", {"num_frames": 1, "selected_frames": [7]})

    model_path = tmp_path / "OCTREE-ANYGS" / drive / "latest"
    model_path.mkdir(parents=True)
    (model_path / "config.yaml").write_text("model: test\n", encoding="utf-8")

    manifest = bundle_run_outputs(
        drive=drive,
        run_output_dir=run_output_dir,
        points_root=tmp_path / "points_world",
        bucket_root=bucket_root,
        colmap_root=colmap_root,
        octree_output_root=tmp_path / "OCTREE-ANYGS",
        model_path=model_path,
        map_viz_output_dir=None,
        render_output_dir=None,
        nbv_output_dir=None,
        viewer_export_iteration=-1,
        viewer_export_output_dir=None,
        viewer_export_archive_path=None,
        skip_local_viewer_export=True,
    )

    assert any(path.endswith("points_world.ply") for path in manifest["missing_optional_artifacts"])
    assert any(path.endswith("uncertainty_histogram.png") for path in manifest["missing_optional_artifacts"])
    assert any(path.endswith("results.json") for path in manifest["missing_optional_artifacts"])
    assert any(path.endswith("per_view.json") for path in manifest["missing_optional_artifacts"])


def test_bundle_writes_single_renderable_archive_with_local_viewer_export(tmp_path):
    scene = "scene_001"
    points_root = tmp_path / "data" / "points_world"
    bucket_root = tmp_path / "data" / "m4" / scene
    colmap_root = tmp_path / "COLMAP"
    octree_root = tmp_path / "OCTREE-ANYGS"
    run_output_dir = tmp_path / "outputs" / "v1_0" / scene
    model_path = octree_root / scene / "run_a"

    write_text(points_root / scene / "points_world.npz", "npz")
    write_text(points_root / scene / "points_world.ply", "ply")
    write_json(
        points_root / scene / "points_world_metadata.json",
        {"dataset": "nvidia_ncore", "point_source": "lidar", "num_frames": 2, "num_points": 10},
    )

    for name in ("U.npy", "uncertainty_components.npz"):
        write_text(bucket_root / name, name)
    write_json(bucket_root / "uncertainty_metadata.json", {"scene": scene})

    write_json(
        colmap_root / scene / "metadata.json",
        {"num_frames": 1, "selected_frames": [100]},
    )
    write_text(colmap_root / scene / "images" / "000001.png", "image")
    write_text(
        colmap_root / scene / "sparse" / "0" / "images.txt",
        "1 1 0 0 0 0 0 0 1 000001.png\n",
    )

    write_text(
        model_path / "config.yaml",
        "model_params:\n  model_config:\n    kwargs:\n      gs_attr: explicit\n",
    )
    write_text(model_path / "point_cloud" / "iteration_42" / "point_cloud_anchor.ply", "ply")

    manifest = bundle_run_outputs(
        drive=scene,
        run_output_dir=run_output_dir,
        points_root=points_root,
        bucket_root=bucket_root,
        colmap_root=colmap_root,
        octree_output_root=octree_root,
        model_path=model_path,
        map_viz_output_dir=None,
        render_output_dir=None,
        nbv_output_dir=None,
        viewer_export_iteration=42,
        viewer_export_output_dir=None,
        viewer_export_archive_path=None,
        skip_local_viewer_export=False,
    )

    run_manifest = json.loads((run_output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    local_viewer = run_manifest["local_viewer_export"]

    assert manifest["archive"]["path"] == str((run_output_dir / f"{scene}.zip").resolve())
    assert local_viewer["output_dir"] == str((run_output_dir / "local_viewer").resolve())
    assert "archive_path" not in local_viewer
    assert local_viewer["viewer_commands"] == str((run_output_dir / "local_viewer" / "VIEWER_COMMANDS.md").resolve())
    assert local_viewer["source_paths"]["octree_model_path"] == str(model_path.resolve())
    assert local_viewer["iteration"] == 42

    assert Path(manifest["archive"]["path"]).is_file()
    assert not (run_output_dir.parent / f"{scene}-local-viewer.zip").exists()
    assert (run_output_dir / "local_viewer" / "model" / "config.yaml").is_file()
    assert (run_output_dir / "local_viewer" / "uncertainty" / "U.npy").is_file()

    with zipfile.ZipFile(manifest["archive"]["path"]) as archive:
        names = set(archive.namelist())
    assert f"{scene}/run_manifest.json" in names
    assert f"{scene}/local_viewer/VIEWER_COMMANDS.md" in names
    assert f"{scene}/local_viewer/model/config.yaml" in names
    assert f"{scene}/local_viewer/model/point_cloud/iteration_42/point_cloud_anchor.ply" in names
    assert f"{scene}/local_viewer/prepared/images/000001.png" not in names
    assert f"{scene}/local_viewer/prepared/sparse/0/images.txt" in names
    assert f"{scene}/local_viewer/uncertainty/U.npy" in names
    assert f"{scene}/{scene}.zip" not in names
