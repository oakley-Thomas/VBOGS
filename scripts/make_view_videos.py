#!/usr/bin/env python3

"""Encode one MP4 per image subdirectory under outputs/views."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.photos_zip_to_video import (
    DEFAULT_EXTENSIONS,
    build_ffmpeg_command,
    natural_sort_key,
    parse_extensions,
    parse_size,
    probe_image_size,
    resolve_executable,
    write_concat_file,
)


DEFAULT_ROOT = REPO_ROOT / "outputs" / "views"


def collect_frame_dirs(root: Path, *, exclude_dirs: set[Path] | None = None) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Views directory not found: {root}")
    resolved_excludes = {path.resolve() for path in exclude_dirs or set()}
    frame_dirs = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.resolve() not in resolved_excludes
    ]
    frame_dirs.sort(key=natural_sort_key)
    if not frame_dirs:
        raise ValueError(f"No subdirectories found under {root}")
    return frame_dirs


def collect_images(frame_dir: Path, extensions: set[str], *, recursive: bool) -> list[Path]:
    paths = frame_dir.rglob("*") if recursive else frame_dir.iterdir()
    images = [
        path.resolve()
        for path in paths
        if path.is_file() and path.suffix.casefold() in extensions
    ]
    images.sort(key=natural_sort_key)
    if not images:
        formatted = ", ".join(sorted(extensions))
        raise ValueError(f"No images with extensions {formatted} were found in {frame_dir}")
    return images


def resolve_video_size(size: str, first_image: Path, ffprobe: str) -> tuple[int, int]:
    parsed_size = parse_size(size)
    if parsed_size is not None:
        return parsed_size
    return probe_image_size(first_image, ffprobe)


def make_view_video(
    frame_dir: Path,
    *,
    output_dir: Path,
    fps: float,
    seconds_per_image: float | None,
    size: str,
    fit: str,
    background: str,
    extensions: set[str],
    recursive: bool,
    overwrite: bool,
    dry_run: bool,
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, object]:
    if fps <= 0:
        raise ValueError("--fps must be positive.")

    frame_duration = seconds_per_image if seconds_per_image is not None else 1.0 / fps
    if frame_duration <= 0:
        raise ValueError("--seconds-per-image must be positive.")

    image_paths = collect_images(frame_dir, extensions, recursive=recursive)
    output_video = output_dir / f"{frame_dir.name}.mp4"
    if output_video.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output already exists: {output_video}. Use --overwrite to replace it.")

    if dry_run:
        return {
            "frame_dir": str(frame_dir),
            "output_video": str(output_video),
            "image_count": len(image_paths),
            "fps": fps,
            "seconds_per_image": frame_duration,
            "dry_run": True,
        }

    resolved_ffmpeg = resolve_executable(ffmpeg)
    resolved_ffprobe = resolve_executable(ffprobe) if size.casefold() == "auto" else ffprobe
    width, height = resolve_video_size(size, image_paths[0], resolved_ffprobe)

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="view_video_") as temp_dir:
        concat_path = Path(temp_dir) / f"{frame_dir.name}-frames.txt"
        write_concat_file(image_paths, concat_path, frame_duration)
        command = build_ffmpeg_command(
            ffmpeg=resolved_ffmpeg,
            concat_path=concat_path,
            output_video=output_video.resolve(),
            width=width,
            height=height,
            fit=fit,
            fps=fps,
            background=background,
            overwrite=overwrite,
        )
        subprocess.run(command, check=True)

    return {
        "frame_dir": str(frame_dir),
        "output_video": str(output_video.resolve()),
        "image_count": len(image_paths),
        "fps": fps,
        "seconds_per_image": frame_duration,
        "size": f"{width}x{height}",
        "dry_run": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory containing one subdirectory per video. Default: outputs/views.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for MP4 files. Defaults to --root.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output video frame rate.")
    parser.add_argument(
        "--seconds-per-image",
        "--seconds-per-photo",
        dest="seconds_per_image",
        type=float,
        default=None,
        help="How long each image should stay on screen. Defaults to one frame per image.",
    )
    parser.add_argument(
        "--size",
        default="auto",
        help="Output size as WIDTHxHEIGHT. Use `auto` to match each subdirectory's first image.",
    )
    parser.add_argument(
        "--fit",
        choices=("contain", "cover", "stretch"),
        default="contain",
        help="How images should fit the output size.",
    )
    parser.add_argument("--background", default="black", help="Padding color for --fit contain.")
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated image extensions to include.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include images from nested directories inside each subfolder.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing videos.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned videos without writing.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue encoding later subdirectories after a failure.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable to use.")
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="ffprobe executable to use when --size auto is selected.",
    )
    return parser


def print_summary(summary: dict[str, object]) -> None:
    action = "Would write" if summary["dry_run"] else "Wrote"
    size = f" ({summary['size']})" if "size" in summary else ""
    print(
        f"{action} {summary['output_video']} from {summary['image_count']} images "
        f"at {summary['fps']:g} fps{size}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or args.root

    try:
        extensions = parse_extensions(args.extensions)
        frame_dirs = collect_frame_dirs(args.root, exclude_dirs={output_dir})
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    failures: list[tuple[Path, Exception]] = []
    for frame_dir in frame_dirs:
        try:
            summary = make_view_video(
                frame_dir,
                output_dir=output_dir,
                fps=args.fps,
                seconds_per_image=args.seconds_per_image,
                size=args.size,
                fit=args.fit,
                background=args.background,
                extensions=extensions,
                recursive=args.recursive,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
            )
        except subprocess.CalledProcessError as exc:
            if not args.continue_on_error:
                print(f"error: {frame_dir}: ffmpeg exited {exc.returncode}", file=sys.stderr)
                raise SystemExit(exc.returncode) from exc
            failures.append((frame_dir, exc))
            print(f"error: {frame_dir}: ffmpeg exited {exc.returncode}", file=sys.stderr, flush=True)
            continue
        except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
            if not args.continue_on_error:
                print(f"error: {exc}", file=sys.stderr)
                raise SystemExit(1) from exc
            failures.append((frame_dir, exc))
            print(f"error: {frame_dir}: {exc}", file=sys.stderr, flush=True)
            continue

        print_summary(summary)

    if failures:
        print("\nFailed view videos:", file=sys.stderr)
        for frame_dir, exc in failures:
            print(f"  - {frame_dir}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
