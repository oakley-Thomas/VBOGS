#!/usr/bin/env python3
"""Extract a zip of photos and encode them into a video with ffmpeg."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence


DEFAULT_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".heif",
)

COMMON_EXECUTABLE_DIRS = (
    Path("/usr/bin"),
    Path("/usr/local/bin"),
    Path("/opt/homebrew/bin"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path, help="Zip file containing photos.")
    parser.add_argument(
        "output_video",
        type=Path,
        nargs="?",
        help="Output video path. Defaults to the zip path with an `.mp4` suffix.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video frame rate.",
    )
    parser.add_argument(
        "--seconds-per-photo",
        type=float,
        default=None,
        help=(
            "How long each photo should stay on screen. "
            "Defaults to one output frame per photo."
        ),
    )
    parser.add_argument(
        "--size",
        default="auto",
        help=(
            "Output size as WIDTHxHEIGHT. Use `auto` to match the first photo. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--fit",
        choices=("contain", "cover", "stretch"),
        default="contain",
        help="How photos should fit the output size.",
    )
    parser.add_argument(
        "--background",
        default="black",
        help="Padding color used with `--fit contain`.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=None,
        help="Directory to extract photos into. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated image extensions to include.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output video if it already exists.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable to use.",
    )
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="ffprobe executable to use when `--size auto` is selected.",
    )
    return parser.parse_args()


def natural_sort_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.as_posix())
    return tuple(int(part) if part.isdigit() else part.casefold() for part in parts)


def parse_extensions(raw: str) -> set[str]:
    extensions = set()
    for value in raw.split(","):
        value = value.strip().casefold()
        if not value:
            continue
        extensions.add(value if value.startswith(".") else f".{value}")
    if not extensions:
        raise ValueError("At least one image extension is required.")
    return extensions


def ensure_inside_directory(path: Path, directory: Path) -> None:
    resolved_path = path.resolve()
    resolved_directory = directory.resolve()
    if resolved_path == resolved_directory:
        return
    if not resolved_path.is_relative_to(resolved_directory):
        raise ValueError(f"Unsafe zip member path: {path}")


def extract_zip(zip_path: Path, extract_dir: Path) -> list[Path]:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_files: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = extract_dir / member.filename
            ensure_inside_directory(target, extract_dir)
            archive.extract(member, extract_dir)
            extracted_files.append(target.resolve())
    return extracted_files


def collect_images(paths: Sequence[Path], extensions: set[str]) -> list[Path]:
    images = [
        path.resolve()
        for path in paths
        if path.is_file() and path.suffix.casefold() in extensions
    ]
    images.sort(key=natural_sort_key)
    if not images:
        formatted = ", ".join(sorted(extensions))
        raise ValueError(f"No images with extensions {formatted} were found in the zip.")
    return images


def parse_size(raw: str) -> tuple[int, int] | None:
    if raw.casefold() == "auto":
        return None
    match = re.fullmatch(r"(\d+)x(\d+)", raw.strip().casefold())
    if not match:
        raise ValueError("Expected --size to be `auto` or WIDTHxHEIGHT, for example 1920x1080.")
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("Output width and height must be positive.")
    if width % 2 or height % 2:
        raise ValueError("Output width and height must be even for H.264 video.")
    return width, height


def probe_image_size(image_path: Path, ffprobe: str) -> tuple[int, int]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(image_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    match = re.search(r"(\d+)x(\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not read image dimensions with ffprobe: {image_path}")
    width = int(match.group(1))
    height = int(match.group(2))
    return width - (width % 2), height - (height % 2)


def escape_concat_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace("'", "\\'")


def write_concat_file(image_paths: Sequence[Path], concat_path: Path, seconds_per_photo: float) -> None:
    concat_path.parent.mkdir(parents=True, exist_ok=True)
    with concat_path.open("w", encoding="utf-8") as handle:
        for path in image_paths:
            handle.write(f"file '{escape_concat_path(path)}'\n")
            handle.write(f"duration {seconds_per_photo:.9f}\n")
        handle.write(f"file '{escape_concat_path(image_paths[-1])}'\n")


def ffmpeg_filter(*, width: int, height: int, fit: str, fps: float, background: str) -> str:
    if fit == "contain":
        fit_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={background}"
        )
    elif fit == "cover":
        fit_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
    elif fit == "stretch":
        fit_filter = f"scale={width}:{height}"
    else:
        raise ValueError(f"Unsupported fit mode: {fit}")
    return f"{fit_filter},setsar=1,fps={fps:g},format=yuv420p"


def build_ffmpeg_command(
    *,
    ffmpeg: str,
    concat_path: Path,
    output_video: Path,
    width: int,
    height: int,
    fit: str,
    fps: float,
    background: str,
    overwrite: bool,
) -> list[str]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
    ]
    command.append("-y" if overwrite else "-n")
    command.extend(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vf",
            ffmpeg_filter(width=width, height=height, fit=fit, fps=fps, background=background),
            "-c:v",
            "libx264",
            "-movflags",
            "+faststart",
            str(output_video),
        ]
    )
    return command


def resolve_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is not None:
        return executable

    path = Path(name)
    if path.is_file() and os.access(path, os.X_OK):
        return str(path.resolve())

    if path.parent == Path("."):
        for directory in COMMON_EXECUTABLE_DIRS:
            candidate = directory / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)

    hint = ""
    if name in {"ffmpeg", "ffprobe"}:
        hint = (
            "\nInstall ffmpeg first, for example:\n"
            "  sudo apt update && sudo apt install ffmpeg\n"
            "or, inside a conda environment:\n"
            "  conda install -c conda-forge ffmpeg"
        )
    raise FileNotFoundError(f"Required executable not found on PATH: {name}{hint}")


def photos_zip_to_video(
    *,
    zip_path: Path,
    output_video: Path,
    fps: float,
    seconds_per_photo: float | None,
    size: str,
    fit: str,
    background: str,
    extract_dir: Path,
    extensions: set[str],
    overwrite: bool,
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, object]:
    if fps <= 0:
        raise ValueError("--fps must be positive.")
    frame_duration = seconds_per_photo if seconds_per_photo is not None else 1.0 / fps
    if frame_duration <= 0:
        raise ValueError("--seconds-per-photo must be positive.")

    zip_path = zip_path.resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    output_video = output_video.resolve()
    if output_video.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_video}. Use --overwrite to replace it.")

    ffmpeg = resolve_executable(ffmpeg)
    if size.casefold() == "auto":
        ffprobe = resolve_executable(ffprobe)

    extracted_files = extract_zip(zip_path, extract_dir.resolve())
    image_paths = collect_images(extracted_files, extensions)

    parsed_size = parse_size(size)
    if parsed_size is None:
        width, height = probe_image_size(image_paths[0], ffprobe)
    else:
        width, height = parsed_size

    output_video.parent.mkdir(parents=True, exist_ok=True)
    concat_path = extract_dir / "_photos_zip_to_video_concat.txt"
    write_concat_file(image_paths, concat_path, frame_duration)
    command = build_ffmpeg_command(
        ffmpeg=ffmpeg,
        concat_path=concat_path,
        output_video=output_video,
        width=width,
        height=height,
        fit=fit,
        fps=fps,
        background=background,
        overwrite=overwrite,
    )
    subprocess.run(command, check=True)

    return {
        "output_video": str(output_video),
        "extract_dir": str(extract_dir.resolve()),
        "image_count": len(image_paths),
        "fps": fps,
        "seconds_per_photo": frame_duration,
        "size": f"{width}x{height}",
    }


def main() -> None:
    args = parse_args()
    output_video = args.output_video or args.zip_path.with_suffix(".mp4")

    try:
        extensions = parse_extensions(args.extensions)
        if args.extract_dir is None:
            with tempfile.TemporaryDirectory(prefix="photos_zip_to_video_") as temp_dir:
                summary = photos_zip_to_video(
                    zip_path=args.zip_path,
                    output_video=output_video,
                    fps=args.fps,
                    seconds_per_photo=args.seconds_per_photo,
                    size=args.size,
                    fit=args.fit,
                    background=args.background,
                    extract_dir=Path(temp_dir),
                    extensions=extensions,
                    overwrite=args.overwrite,
                    ffmpeg=args.ffmpeg,
                    ffprobe=args.ffprobe,
                )
        else:
            summary = photos_zip_to_video(
                zip_path=args.zip_path,
                output_video=output_video,
                fps=args.fps,
                seconds_per_photo=args.seconds_per_photo,
                size=args.size,
                fit=args.fit,
                background=args.background,
                extract_dir=args.extract_dir,
                extensions=extensions,
                overwrite=args.overwrite,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
            )
    except subprocess.CalledProcessError as exc:
        print(f"error: ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        "Wrote {output_video} from {image_count} images "
        "at {fps:g} fps ({seconds_per_photo:g}s/photo, {size}).".format(**summary)
    )


if __name__ == "__main__":
    main()
