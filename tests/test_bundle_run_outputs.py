import json

import numpy as np

from scripts.bundle_run_outputs import bundle_run_outputs


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
    )

    assert (run_output_dir / "pointclouds" / "stereo" / "points_world.npz").exists()
    assert (run_output_dir / "pointclouds" / "stereo" / "points_world.ply").exists()
    assert (run_output_dir / "uncertainty" / "U.npy").exists()
    assert (run_output_dir / "prepared" / "metadata.json").exists()
    assert (run_output_dir / "octree" / "config.yaml").exists()
    assert (run_output_dir / "run_manifest.json").exists()

    saved_manifest = json.loads((run_output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["drive"] == drive
    assert saved_manifest["frame_counts"]["num_frames"] == 1000
    assert saved_manifest["frame_counts"]["selected_frame_count"] == 1000
    assert saved_manifest["stereo"]["num_points"] == 2
    assert saved_manifest["stage_outputs"]["rendered_views"] == str((run_output_dir / "views").resolve())
    assert saved_manifest["source_paths"]["octree_model_path"] == str(model_path.resolve())
    assert manifest["missing_optional_artifacts"] == []


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
    )

    assert any(path.endswith("points_world.ply") for path in manifest["missing_optional_artifacts"])
    assert any(path.endswith("uncertainty_histogram.png") for path in manifest["missing_optional_artifacts"])
