from pathlib import Path

from vbogs.data_layout import resolve_kitti360_path


def test_resolve_raw_layout_prefers_drive_complete_data_2d_test(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    drive = "2013_05_28_drive_0008_sync"

    # An incomplete older root should not win just because the top directory exists.
    (tmp_path / "data" / "KITTI-360" / "images").mkdir(parents=True)
    new_root = tmp_path / "data" / "KITTI-360" / "data_2d_test"
    (new_root / drive / "image_00" / "data_rect").mkdir(parents=True)
    (new_root / drive / "image_01" / "data_rect").mkdir(parents=True)

    resolved = resolve_kitti360_path(None, kind="raw", drive=drive)

    assert resolved == Path("data/KITTI-360/data_2d_test")


def test_resolve_raw_layout_uses_complete_older_images_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    drive = "2013_05_28_drive_0008_sync"
    old_root = tmp_path / "data" / "KITTI-360" / "images"
    (old_root / drive / "image_00" / "data_rect").mkdir(parents=True)
    (old_root / drive / "image_01" / "data_rect").mkdir(parents=True)

    resolved = resolve_kitti360_path(None, kind="raw", drive=drive)

    assert resolved == Path("data/KITTI-360/images")


def test_resolve_explicit_path_without_validation(tmp_path):
    explicit = tmp_path / "custom-kitti"

    resolved = resolve_kitti360_path(explicit, kind="raw", drive="drive")

    assert resolved == explicit
