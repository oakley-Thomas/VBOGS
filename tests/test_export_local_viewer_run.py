import json
import zipfile

import numpy as np
import yaml

from scripts.export_local_viewer_run import export_local_viewer_run
from vbogs.octree_config import resolve_relative_source_path


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_resolve_relative_source_path_is_model_relative(tmp_path):
    model_path = tmp_path / "export" / "model"
    cfg = {"model_params": {"source_path": "../prepared"}}

    resolved = resolve_relative_source_path(cfg, model_path)

    assert resolved["model_params"]["source_path"] == str((tmp_path / "export" / "prepared").resolve())
    assert cfg["model_params"]["source_path"] == "../prepared"


def test_export_local_viewer_run_copies_renderable_layout_and_archive(tmp_path):
    drive = "drive_sync"
    source_model = tmp_path / "OCTREE-ANYGS" / drive / "run-a"
    checkpoint = source_model / "point_cloud" / "iteration_7"
    checkpoint.mkdir(parents=True)
    (checkpoint / "point_cloud_anchor.ply").write_bytes(b"ply\n")
    (checkpoint / "opacity_mlp.pt").write_bytes(b"opacity")
    (checkpoint / "cov_mlp.pt").write_bytes(b"cov")
    (checkpoint / "color_mlp.pt").write_bytes(b"color")
    (source_model / "input.ply").write_bytes(b"input")
    write_yaml(
        source_model / "config.yaml",
        {
            "model_params": {
                "source_path": "/data/COLMAP/drive_sync",
                "model_config": {
                    "kwargs": {
                        "gs_attr": "implicit3D",
                        "appearance_dim": 0,
                    }
                },
            },
            "optim_params": {"iterations": 7},
            "pipeline_params": {},
        },
    )

    raw_images = tmp_path / "raw_images"
    raw_images.mkdir()
    (raw_images / "000000.png").write_bytes(b"raw-png")
    colmap_scene = tmp_path / "COLMAP" / drive
    (colmap_scene / "images").mkdir(parents=True)
    (colmap_scene / "images" / "000000.png").symlink_to(raw_images / "000000.png")
    (colmap_scene / "sparse" / "0").mkdir(parents=True)
    (colmap_scene / "sparse" / "0" / "cameras.txt").write_text("camera\n", encoding="utf-8")
    (colmap_scene / "sparse" / "0" / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 000000.png\n0.0 0.0 -1\n",
        encoding="utf-8",
    )
    (colmap_scene / "sparse" / "0" / "points3D.ply").write_bytes(b"ply\n")
    (colmap_scene / "metadata.json").write_text("{}", encoding="utf-8")

    bucket_root = tmp_path / "m4" / drive
    bucket_root.mkdir(parents=True)
    np.save(bucket_root / "U.npy", np.array([1.0, 2.0], dtype=np.float32))
    (bucket_root / "uncertainty_metadata.json").write_text('{"anchor_count": 2}', encoding="utf-8")

    output_dir = tmp_path / "exports" / drive
    archive_path = tmp_path / "exports" / f"{drive}-local-viewer.zip"
    manifest = export_local_viewer_run(
        drive=drive,
        model_path=source_model,
        octree_output_root=tmp_path / "OCTREE-ANYGS",
        colmap_root=tmp_path / "COLMAP",
        colmap_path=None,
        bucket_root=bucket_root,
        u_path=None,
        iteration=-1,
        output_dir=output_dir,
        archive_path=archive_path,
        write_archive=True,
        overwrite=False,
    )

    assert (output_dir / "model" / "config.yaml").is_file()
    assert (output_dir / "model" / "original_config.yaml").is_file()
    assert (output_dir / "model" / "point_cloud" / "iteration_7" / "point_cloud_anchor.ply").is_file()
    assert (output_dir / "prepared" / "sparse" / "0" / "images.txt").is_file()
    assert (output_dir / "prepared" / "images" / "000000.png").is_file()
    assert not (output_dir / "prepared" / "images" / "000000.png").is_symlink()
    assert (output_dir / "prepared" / "images" / "000000.png").read_bytes() == b"raw-png"
    assert (output_dir / "uncertainty" / "U.npy").is_file()
    assert (output_dir / "VIEWER_COMMANDS.md").is_file()
    assert archive_path.is_file()

    patched_cfg = yaml.safe_load((output_dir / "model" / "config.yaml").read_text(encoding="utf-8"))
    assert patched_cfg["model_params"]["source_path"] == "../prepared"
    assert manifest["iteration"] == 7
    assert manifest["local_viewer"]["config_source_path"] == "../prepared"
    assert manifest["prepared_images"]["image_count"] == 1
    assert manifest["prepared_images"]["materialized"] is True

    saved_manifest = json.loads((output_dir / "local_viewer_manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["archive_path"] == str(archive_path.resolve())

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert f"{drive}/model/config.yaml" in names
    assert f"{drive}/prepared/images/000000.png" in names
    assert f"{drive}/prepared/sparse/0/cameras.txt" in names
    assert f"{drive}/uncertainty/U.npy" in names
    assert f"{drive}/local_viewer_manifest.json" in names
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.read(f"{drive}/prepared/images/000000.png") == b"raw-png"
