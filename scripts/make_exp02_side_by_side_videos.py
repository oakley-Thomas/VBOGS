#!/usr/bin/env python3

"""Encode exp02 side-by-side render frames into per-drive videos."""

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

from scripts.package_exp02_artifacts import VariantSpec, build_variant_specs, drive_names_for_spec
from scripts.photos_zip_to_video import (
    build_ffmpeg_command,
    natural_sort_key,
    parse_extensions,
    parse_size,
    probe_image_size,
    resolve_executable,
    write_concat_file,
)


SPLITS = ("train", "test")
DEFAULT_EXTENSIONS = ".png"


def split_side_by_side_dir(spec: VariantSpec, drive: str, split: str) -> Path:
    return spec.root / drive / "views" / split / "side_by_side"


def output_video_path(spec: VariantSpec, drive: str, split: str) -> Path:
    return spec.root / f"{drive}-{spec.name}-{split}.mp4"


def collect_side_by_side_images(side_by_side_dir: Path, extensions: set[str]) -> list[Path]:
    if not side_by_side_dir.is_dir():
        raise FileNotFoundError(f"Side-by-side image directory not found: {side_by_side_dir}")
    images = [
        path.resolve()
        for path in side_by_side_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in extensions
    ]
    images.sort(key=natural_sort_key)
    if not images:
        formatted = ", ".join(sorted(extensions))
        raise ValueError(f"No images with extensions {formatted} were found in {side_by_side_dir}")
    return images


def resolve_video_size(size: str, first_image: Path, ffprobe: str) -> tuple[int, int]:
    parsed_size = parse_size(size)
    if parsed_size is not None:
        return parsed_size
    return probe_image_size(first_image, ffprobe)


def make_split_video(
    spec: VariantSpec,
    drive: str,
    split: str,
    *,
    fps: float,
    seconds_per_photo: float | None,
    size: str,
    fit: str,
    background: str,
    extensions: set[str],
    overwrite: bool,
    dry_run: bool,
    ffmpeg: str,
    ffprobe: str,
) -> dict[str, object]:
    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    if fps <= 0:
        raise ValueError("--fps must be positive.")

    frame_duration = seconds_per_photo if seconds_per_photo is not None else 1.0 / fps
    if frame_duration <= 0:
        raise ValueError("--seconds-per-photo must be positive.")

    side_by_side_dir = split_side_by_side_dir(spec, drive, split)
    image_paths = collect_side_by_side_images(side_by_side_dir, extensions)
    output_video = output_video_path(spec, drive, split)
    if output_video.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output already exists: {output_video}. Use --overwrite to replace it.")

    if dry_run:
        return {
            "drive": drive,
            "variant": spec.name,
            "split": split,
            "output_video": str(output_video),
            "image_count": len(image_paths),
            "fps": fps,
            "seconds_per_photo": frame_duration,
            "dry_run": True,
        }

    resolved_ffmpeg = resolve_executable(ffmpeg)
    resolved_ffprobe = resolve_executable(ffprobe) if size.casefold() == "auto" else ffprobe
    width, height = resolve_video_size(size, image_paths[0], resolved_ffprobe)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="exp02_side_by_side_video_") as temp_dir:
        concat_path = Path(temp_dir) / f"{drive}-{spec.name}-{split}-frames.txt"
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
        "drive": drive,
        "variant": spec.name,
        "split": split,
        "output_video": str(output_video.resolve()),
        "image_count": len(image_paths),
        "fps": fps,
        "seconds_per_photo": frame_duration,
        "size": f"{width}x{height}",
        "dry_run": False,
    }


def make_drive_videos(
    spec: VariantSpec,
    drive: str,
    *,
    fps: float,
    seconds_per_photo: float | None,
    size: str,
    fit: str,
    background: str,
    extensions: set[str],
    overwrite: bool,
    dry_run: bool,
    ffmpeg: str,
    ffprobe: str,
) -> list[dict[str, object]]:
    return [
        make_split_video(
            spec,
            drive,
            split,
            fps=fps,
            seconds_per_photo=seconds_per_photo,
            size=size,
            fit=fit,
            background=background,
            extensions=extensions,
            overwrite=overwrite,
            dry_run=dry_run,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
        for split in SPLITS
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("explicit", "implicit", "both"),
        default="both",
        help="Exp02 variant to encode.",
    )
    parser.add_argument(
        "--drive",
        action="append",
        dest="drives",
        default=None,
        help="Drive id to encode. May be repeated. Defaults to all drives in selected roots.",
    )
    parser.add_argument("--explicit-root", type=Path, default=None)
    parser.add_argument("--implicit-root", type=Path, default=None)
    parser.add_argument("--explicit-config", type=Path, default=None)
    parser.add_argument("--implicit-config", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=10.0, help="Output video frame rate.")
    parser.add_argument(
        "--seconds-per-photo",
        type=float,
        default=None,
        help="How long each image should stay on screen. Defaults to one frame per image.",
    )
    parser.add_argument(
        "--size",
        default="auto",
        help="Output size as WIDTHxHEIGHT. Use `auto` to match the first image.",
    )
    parser.add_argument(
        "--fit",
        choices=("contain", "cover", "stretch"),
        default="contain",
        help="How images should fit the output size.",
    )
    parser.add_argument("--background", default="black", help="Padding color for --fit contain.")
    parser.add_argument("--extensions", default=DEFAULT_EXTENSIONS)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing videos.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned videos without writing.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue encoding later drives after a failure.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable to use.")
    parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        help="ffprobe executable to use when --size auto is selected.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    extensions = parse_extensions(args.extensions)
    failures: list[tuple[str, str, Exception]] = []

    for spec in build_variant_specs(args):
        drives = drive_names_for_spec(spec, args.drives)
        for drive in drives:
            try:
                summaries = make_drive_videos(
                    spec,
                    drive,
                    fps=args.fps,
                    seconds_per_photo=args.seconds_per_photo,
                    size=args.size,
                    fit=args.fit,
                    background=args.background,
                    extensions=extensions,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    ffmpeg=args.ffmpeg,
                    ffprobe=args.ffprobe,
                )
            except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
                if not args.continue_on_error:
                    raise SystemExit(f"error: {exc}") from exc
                failures.append((spec.name, drive, exc))
                print(f"error: {drive} ({spec.name}): {exc}", file=sys.stderr, flush=True)
                continue
            except subprocess.CalledProcessError as exc:
                if not args.continue_on_error:
                    raise SystemExit(exc.returncode) from exc
                failures.append((spec.name, drive, exc))
                print(
                    f"error: {drive} ({spec.name}): ffmpeg exited {exc.returncode}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            action = "Would write" if args.dry_run else "Wrote"
            for summary in summaries:
                print(
                    f"{action} {summary['output_video']} from "
                    f"{summary['image_count']} {summary['split']} images at {summary['fps']:g} fps",
                    flush=True,
                )

    if failures:
        print("\nFailed exp02 videos:", file=sys.stderr)
        for variant, drive, exc in failures:
            print(f"  - {drive} ({variant}): {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
