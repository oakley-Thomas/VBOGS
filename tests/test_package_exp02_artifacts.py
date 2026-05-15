import argparse
import zipfile
from pathlib import Path

import pytest

from scripts.package_exp02_artifacts import (
    DEFAULT_CONFIGS,
    DEFAULT_ROOTS,
    VariantSpec,
    build_variant_specs,
    package_drive_artifacts,
)


def make_exp02_drive(root: Path, drive: str) -> None:
    for relative_dir in (
        Path("views") / "train" / "side_by_side",
        Path("views") / "test" / "side_by_side",
        Path("pointclouds") / "anchors",
    ):
        (root / drive / relative_dir).mkdir(parents=True)
    (root / drive / "views" / "train" / "side_by_side" / "00002.png").write_bytes(b"train2")
    (root / drive / "views" / "test" / "side_by_side" / "00001.png").write_bytes(b"test1")
    (root / drive / "pointclouds" / "anchors" / "anchors_uncertainty_all.ply").write_bytes(b"ply")
    (root / drive / "pointclouds" / "anchors" / "uncertainty_map_metadata.json").write_text(
        "{}",
        encoding="utf-8",
    )


def test_package_drive_artifacts_writes_variant_named_zip_with_curated_members(tmp_path):
    drive = "2013_05_28_drive_0000_sync"
    root = tmp_path / "exp02_explicit3d_baseline"
    config = tmp_path / "exp02-A_explicit3d_baseline_pipeline_config.yaml"
    config.write_text("pipeline:\n  drive: test\n", encoding="utf-8")
    make_exp02_drive(root, drive)

    summary = package_drive_artifacts(
        VariantSpec("explicit", root, config),
        drive,
        overwrite=False,
        dry_run=False,
    )

    archive_path = root / f"{drive}-explicit.zip"
    assert summary["archive_path"] == str(archive_path.resolve())
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())

    top = f"{drive}-explicit"
    assert f"{top}/pipeline_config.yaml" in names
    assert f"{top}/views/train/side_by_side/00002.png" in names
    assert f"{top}/views/test/side_by_side/00001.png" in names
    assert f"{top}/pointclouds/anchors/anchors_uncertainty_all.ply" in names
    assert f"{top}/pointclouds/anchors/uncertainty_map_metadata.json" in names


def test_package_drive_artifacts_requires_curated_dirs(tmp_path):
    drive = "2013_05_28_drive_0000_sync"
    root = tmp_path / "exp02_implicit3d_baseline"
    config = tmp_path / "exp02_B_implicit3d_baseline_pipeline_config.yaml"
    config.write_text("pipeline: {}\n", encoding="utf-8")
    (root / drive / "views" / "train" / "side_by_side").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Required artifact directory"):
        package_drive_artifacts(
            VariantSpec("implicit", root, config),
            drive,
            overwrite=False,
            dry_run=False,
        )


def test_package_drive_artifacts_overwrite_controls_existing_zip(tmp_path):
    drive = "2013_05_28_drive_0000_sync"
    root = tmp_path / "exp02_explicit3d_baseline"
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}\n", encoding="utf-8")
    make_exp02_drive(root, drive)
    archive_path = root / f"{drive}-explicit.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("old.txt", "old")

    with pytest.raises(FileExistsError):
        package_drive_artifacts(
            VariantSpec("explicit", root, config),
            drive,
            overwrite=False,
            dry_run=False,
        )

    package_drive_artifacts(
        VariantSpec("explicit", root, config),
        drive,
        overwrite=True,
        dry_run=False,
    )
    with zipfile.ZipFile(archive_path) as archive:
        assert "old.txt" not in archive.namelist()


def test_build_variant_specs_defaults_to_canonical_roots_and_configs():
    args = argparse.Namespace(
        variant="both",
        explicit_root=None,
        implicit_root=None,
        explicit_config=None,
        implicit_config=None,
    )

    specs = build_variant_specs(args)

    assert specs == [
        VariantSpec("explicit", DEFAULT_ROOTS["explicit"], DEFAULT_CONFIGS["explicit"]),
        VariantSpec("implicit", DEFAULT_ROOTS["implicit"], DEFAULT_CONFIGS["implicit"]),
    ]
