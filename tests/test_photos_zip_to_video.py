import zipfile

import pytest

from scripts.photos_zip_to_video import (
    build_ffmpeg_command,
    collect_images,
    extract_zip,
    parse_size,
    write_concat_file,
)


def test_extract_zip_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "photos.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.jpg", b"not really an image")

    with pytest.raises(ValueError, match="Unsafe zip member path"):
        extract_zip(archive_path, tmp_path / "extracted")


def test_collect_images_uses_natural_sort(tmp_path):
    paths = []
    for name in ("frame10.jpg", "frame2.jpg", "frame1.jpg", "notes.txt"):
        path = tmp_path / name
        path.write_bytes(b"x")
        paths.append(path)

    images = collect_images(paths, {".jpg"})

    assert [path.name for path in images] == ["frame1.jpg", "frame2.jpg", "frame10.jpg"]


def test_parse_size_requires_even_dimensions():
    assert parse_size("1920x1080") == (1920, 1080)
    assert parse_size("auto") is None

    with pytest.raises(ValueError, match="even"):
        parse_size("1919x1080")


def test_write_concat_file_repeats_last_image(tmp_path):
    images = [tmp_path / "a.jpg", tmp_path / "b's.jpg"]
    concat_path = tmp_path / "frames.txt"
    escaped_second = str(images[1]).replace("'", "\\'")

    write_concat_file(images, concat_path, 0.5)

    assert concat_path.read_text(encoding="utf-8").splitlines() == [
        f"file '{images[0]}'",
        "duration 0.500000000",
        f"file '{escaped_second}'",
        "duration 0.500000000",
        f"file '{escaped_second}'",
    ]


def test_build_ffmpeg_command_sets_video_options(tmp_path):
    command = build_ffmpeg_command(
        ffmpeg="ffmpeg",
        concat_path=tmp_path / "frames.txt",
        output_video=tmp_path / "out.mp4",
        width=1280,
        height=720,
        fit="contain",
        fps=24.0,
        background="black",
        overwrite=True,
    )

    assert command[:2] == ["ffmpeg", "-hide_banner"]
    assert "-y" in command
    assert "-f" in command
    assert "concat" in command
    assert "-c:v" in command
    assert "libx264" in command
    assert any("scale=1280:720:force_original_aspect_ratio=decrease" in part for part in command)
