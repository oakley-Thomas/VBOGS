from pathlib import Path

import pytest

from scripts import make_exp02_side_by_side_videos as videos
from scripts.package_exp02_artifacts import VariantSpec
from scripts.photos_zip_to_video import parse_extensions


def make_side_by_side_images(root: Path, drive: str) -> None:
    for split in ("train", "test"):
        split_dir = root / drive / "views" / split / "side_by_side"
        split_dir.mkdir(parents=True)
        for name in ("frame10.png", "frame2.png", "frame1.png"):
            (split_dir / name).write_bytes(b"png")


def test_parser_defaults_to_10_fps():
    args = videos.build_parser().parse_args([])

    assert args.fps == 10.0


def test_make_split_video_uses_natural_sort_and_variant_output_name(tmp_path, monkeypatch):
    drive = "2013_05_28_drive_0000_sync"
    root = tmp_path / "exp02_explicit3d_baseline"
    make_side_by_side_images(root, drive)
    spec = VariantSpec("explicit", root, tmp_path / "config.yaml")
    commands: list[list[str]] = []
    concat_lines: list[str] = []

    monkeypatch.setattr(videos, "resolve_executable", lambda name: name)
    monkeypatch.setattr(videos, "probe_image_size", lambda image_path, ffprobe: (1280, 720))

    def fake_run(command, check):
        assert check is True
        commands.append(command)
        concat_path = Path(command[command.index("-i") + 1])
        concat_lines.extend(concat_path.read_text(encoding="utf-8").splitlines())

    monkeypatch.setattr(videos.subprocess, "run", fake_run)

    summary = videos.make_split_video(
        spec,
        drive,
        "train",
        fps=10.0,
        seconds_per_photo=None,
        size="auto",
        fit="contain",
        background="black",
        extensions=parse_extensions(".png"),
        overwrite=False,
        dry_run=False,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
    )

    assert summary["output_video"] == str((root / f"{drive}-explicit-train.mp4").resolve())
    assert summary["image_count"] == 3
    assert len(commands) == 1
    assert any("fps=10" in part for part in commands[0])
    frame_lines = [line for line in concat_lines if line.startswith("file ")]
    assert frame_lines[0].endswith("frame1.png'")
    assert frame_lines[1].endswith("frame2.png'")
    assert frame_lines[2].endswith("frame10.png'")


def test_make_drive_videos_writes_train_and_test_outputs(tmp_path, monkeypatch):
    drive = "2013_05_28_drive_0000_sync"
    root = tmp_path / "exp02_implicit3d_baseline"
    make_side_by_side_images(root, drive)
    spec = VariantSpec("implicit", root, tmp_path / "config.yaml")

    monkeypatch.setattr(videos, "resolve_executable", lambda name: name)
    monkeypatch.setattr(videos, "probe_image_size", lambda image_path, ffprobe: (640, 480))
    monkeypatch.setattr(videos.subprocess, "run", lambda command, check: None)

    summaries = videos.make_drive_videos(
        spec,
        drive,
        fps=10.0,
        seconds_per_photo=None,
        size="auto",
        fit="contain",
        background="black",
        extensions=parse_extensions(".png"),
        overwrite=False,
        dry_run=False,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
    )

    assert [summary["split"] for summary in summaries] == ["train", "test"]
    assert [Path(summary["output_video"]).name for summary in summaries] == [
        f"{drive}-implicit-train.mp4",
        f"{drive}-implicit-test.mp4",
    ]


def test_make_drive_videos_dry_run_skips_ffmpeg(tmp_path, monkeypatch):
    drive = "2013_05_28_drive_0000_sync"
    root = tmp_path / "exp02_explicit3d_baseline"
    make_side_by_side_images(root, drive)
    spec = VariantSpec("explicit", root, tmp_path / "config.yaml")

    def fail_run(command, check):
        raise AssertionError("ffmpeg should not run in dry-run mode")

    monkeypatch.setattr(videos.subprocess, "run", fail_run)

    summaries = videos.make_drive_videos(
        spec,
        drive,
        fps=10.0,
        seconds_per_photo=None,
        size="auto",
        fit="contain",
        background="black",
        extensions=parse_extensions(".png"),
        overwrite=False,
        dry_run=True,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
    )

    assert len(summaries) == 2
    assert all(summary["dry_run"] is True for summary in summaries)


def test_make_split_video_respects_overwrite(tmp_path):
    drive = "2013_05_28_drive_0000_sync"
    root = tmp_path / "exp02_explicit3d_baseline"
    make_side_by_side_images(root, drive)
    (root / f"{drive}-explicit-train.mp4").write_bytes(b"old")
    spec = VariantSpec("explicit", root, tmp_path / "config.yaml")

    with pytest.raises(FileExistsError):
        videos.make_split_video(
            spec,
            drive,
            "train",
            fps=10.0,
            seconds_per_photo=None,
            size="auto",
            fit="contain",
            background="black",
            extensions=parse_extensions(".png"),
            overwrite=False,
            dry_run=False,
            ffmpeg="ffmpeg",
            ffprobe="ffprobe",
        )
