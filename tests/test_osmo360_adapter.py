import argparse
import json

import numpy as np

from scripts.export_points_world import export_osmo360_points
from scripts.prepare_osmo360_colmap import prepare_dataset
from vbogs.osmo360_adapter import read_colmap_points3d_txt, read_ply_xyz_rgb, write_binary_ply


def test_read_colmap_points3d_txt_parses_xyz_rgb(tmp_path):
    points_path = tmp_path / "points3D.txt"
    points_path.write_text(
        """
# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
1 1.0 2.0 3.0 10 20 30 0.1 1 2
2 -1.0 -2.0 -3.0 40 50 60 0.2 2 3
""".lstrip(),
        encoding="utf-8",
    )

    point_cloud = read_colmap_points3d_txt(points_path)

    np.testing.assert_allclose(
        point_cloud.xyz,
        np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float32),
    )
    assert point_cloud.rgb.tolist() == [[10, 20, 30], [40, 50, 60]]


def test_read_and_write_ply_preserves_xyz_rgb(tmp_path):
    ply_path = tmp_path / "points.ply"
    xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    rgb = np.array([[10, 20, 30]], dtype=np.uint8)

    write_binary_ply(ply_path, xyz, rgb)
    point_cloud = read_ply_xyz_rgb(ply_path)

    np.testing.assert_allclose(point_cloud.xyz, xyz)
    assert point_cloud.rgb.tolist() == [[10, 20, 30]]


def test_prepare_osmo360_dry_run_writes_command_metadata(tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"not a real video")
    tool_root = tmp_path / "tool"
    script_path = tool_root / "cli_tools" / "gs360_360PerspCut.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("# placeholder\n", encoding="utf-8")
    args = argparse.Namespace(
        dataset_name="dji_osmo360",
        scene_id="capture01",
        video=video_path,
        fps=0.5,
        perspective_preset="full360coverage",
        matcher="sequential",
        dense=True,
        output_root=tmp_path / "COLMAP",
        tool_root=tool_root,
        ffmpeg="ffmpeg",
        colmap="colmap",
        image_size=800,
        image_ext="jpg",
        start=1.0,
        end=3.0,
        max_image_size=800,
        camera_hfov=None,
        overwrite=True,
        dry_run=True,
    )

    dataset_dir = prepare_dataset(args)
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["dataset"] == "dji_osmo360"
    assert metadata["dry_run"] is True
    assert metadata["perspective_export"]["preset"] == "full360coverage"
    assert metadata["perspective_export"]["fps"] == 0.5
    command_text = "\n".join(command["display"] for command in metadata["commands"])
    assert "gs360_360PerspCut.py" in command_text
    assert "feature_extractor" in command_text
    assert "sequential_matcher" in command_text
    assert "patch_match_stereo" in command_text
    assert "stereo_fusion" in command_text
    assert "--ImageReader.camera_model PINHOLE" in command_text


def test_export_osmo360_dense_points_preserves_contract(tmp_path):
    scene_dir = tmp_path / "COLMAP" / "capture01"
    dense_dir = scene_dir / "dense"
    dense_dir.mkdir(parents=True)
    write_binary_ply(
        dense_dir / "fused.ply",
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.array([[10, 20, 30]], dtype=np.uint8),
    )
    args = argparse.Namespace(
        dataset_name="dji_osmo360",
        scene_id="capture01",
        drive=None,
        output_root=tmp_path / "points_world",
        output_name="points_world.npz",
        write_ply=False,
        colmap_root=tmp_path / "COLMAP",
        dense_ply=None,
        sparse_points=None,
    )

    output_path = export_osmo360_points(args, "mvs_dense")
    payload = np.load(output_path)
    metadata = json.loads(output_path.with_name("points_world_metadata.json").read_text(encoding="utf-8"))

    np.testing.assert_allclose(payload["xyz"], np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert payload["rgb"].tolist() == [[10, 20, 30]]
    assert payload["frame_id"].tolist() == [-1]
    assert metadata["point_source"] == "mvs_dense"


def test_export_osmo360_sparse_points_preserves_contract(tmp_path):
    sparse_dir = tmp_path / "COLMAP" / "capture01" / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    (sparse_dir / "points3D.txt").write_text(
        "1 1.0 2.0 3.0 10 20 30 0.1 1 2\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        dataset_name="dji_osmo360",
        scene_id="capture01",
        drive=None,
        output_root=tmp_path / "points_world",
        output_name="points_world.npz",
        write_ply=False,
        colmap_root=tmp_path / "COLMAP",
        dense_ply=None,
        sparse_points=None,
    )

    output_path = export_osmo360_points(args, "sfm_sparse")
    payload = np.load(output_path)

    np.testing.assert_allclose(payload["xyz"], np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    assert payload["rgb"].tolist() == [[10, 20, 30]]
    assert payload["frame_id"].tolist() == [-1]
