#!/usr/bin/env python3

"""Prepare a DJI Osmo 360 equirectangular video for Octree-AnyGS.

The adapter converts a stitched 360 video into virtual perspective views with
the pinned 360Cam toolkit, runs local COLMAP on those PINHOLE views, and writes
the same COLMAP-style contract used by the other VBOGS dataset adapters:

- images/
- sparse/0/cameras.txt
- sparse/0/images.txt
- sparse/0/points3D.ply
- metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.osmo360_adapter import (
    command_to_string,
    count_colmap_images_txt,
    list_images,
    read_colmap_points3d_txt,
    write_binary_ply,
)


PINNED_360CAM_COMMIT = "4190130296c28cc3f0235058d108f8e8cc904e30"
DEFAULT_TOOL_ROOT = REPO_ROOT / "third_party" / "360Cam-PGM-3DGS-Tools"
PRESET_FOCAL_MM = {
    "default": 12.0,
    "fisheyelike": 17.0,
    "full360coverage": 14.0,
    "2views": 6.0,
    "evenMinus30": 12.0,
    "evenPlus30": 12.0,
    "fisheyeXY": 12.0,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", choices=("dji_osmo360",), default="dji_osmo360")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument(
        "--perspective-preset",
        choices=tuple(PRESET_FOCAL_MM.keys()),
        default="full360coverage",
    )
    parser.add_argument("--matcher", choices=("sequential", "exhaustive"), default="sequential")
    parser.add_argument("--dense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", type=Path, default=Path("/data/COLMAP"))
    parser.add_argument("--tool-root", type=Path, default=DEFAULT_TOOL_ROOT)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--image-size", type=int, default=1600)
    parser.add_argument("--image-ext", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--max-image-size", type=int, default=1600)
    parser.add_argument("--camera-hfov", type=float, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def fov_from_focal_mm(focal_mm: float, sensor_width_mm: float = 36.0) -> float:
    return math.degrees(2.0 * math.atan(sensor_width_mm / (2.0 * focal_mm)))


def pinhole_params(*, width: int, height: int, hfov_deg: float) -> tuple[float, float, float, float]:
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = fx
    return fx, fy, width / 2.0, height / 2.0


def run_command(
    command: Sequence[object],
    *,
    dry_run: bool,
    commands: list[dict[str, Any]],
    cwd: Path | None = None,
) -> None:
    command_list = [str(part) for part in command]
    commands.append(
        {
            "cmd": command_list,
            "cwd": str(cwd) if cwd is not None else None,
            "display": command_to_string(command_list),
        }
    )
    print(command_to_string(command_list))
    if dry_run:
        return
    subprocess.run(command_list, cwd=str(cwd) if cwd is not None else None, check=True)


def require_360cam_tool(tool_root: Path) -> Path:
    script = tool_root / "cli_tools" / "gs360_360PerspCut.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"360Cam perspective cutter not found at {script}. "
            "Initialize the submodule with: git submodule update --init --recursive"
        )
    return script


def build_perspective_command(args: argparse.Namespace, images_dir: Path) -> list[object]:
    script = require_360cam_tool(args.tool_root)
    command: list[object] = [
        sys.executable,
        script,
        "--in",
        args.video,
        "--out",
        images_dir,
        "--preset",
        args.perspective_preset,
        "--size",
        args.image_size,
        "--ext",
        args.image_ext,
        "--fps",
        args.fps,
        "--ffmpeg",
        args.ffmpeg,
        "--print-cmd",
        "once",
    ]
    if args.start is not None:
        command.extend(["--start", args.start])
    if args.end is not None:
        command.extend(["--end", args.end])
    return command


def reset_output_dirs(args: argparse.Namespace, dataset_dir: Path) -> dict[str, Path]:
    images_dir = dataset_dir / "images"
    sparse_dir = dataset_dir / "sparse" / "0"
    colmap_dir = dataset_dir / "colmap"
    dense_dir = dataset_dir / "dense"
    if args.overwrite and dataset_dir.exists() and not args.dry_run:
        shutil.rmtree(dataset_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir.mkdir(parents=True, exist_ok=True)
    if args.dense:
        dense_dir.mkdir(parents=True, exist_ok=True)
    return {
        "images": images_dir,
        "sparse": sparse_dir,
        "colmap": colmap_dir,
        "dense": dense_dir,
    }


def colmap_commands(
    *,
    args: argparse.Namespace,
    images_dir: Path,
    colmap_dir: Path,
    sparse_dir: Path,
    dense_dir: Path,
    hfov_deg: float,
) -> list[list[object]]:
    database = colmap_dir / "database.db"
    raw_sparse = colmap_dir / "sparse"
    raw_sparse.mkdir(parents=True, exist_ok=True)
    fx, fy, cx, cy = pinhole_params(width=args.image_size, height=args.image_size, hfov_deg=hfov_deg)
    camera_params = f"{fx:.8f},{fy:.8f},{cx:.8f},{cy:.8f}"
    commands: list[list[object]] = [
        [
            args.colmap,
            "feature_extractor",
            "--database_path",
            database,
            "--image_path",
            images_dir,
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.camera_params",
            camera_params,
        ]
    ]
    if args.matcher == "sequential":
        commands.append([args.colmap, "sequential_matcher", "--database_path", database])
    else:
        commands.append([args.colmap, "exhaustive_matcher", "--database_path", database])
    commands.extend(
        [
            [
                args.colmap,
                "mapper",
                "--database_path",
                database,
                "--image_path",
                images_dir,
                "--output_path",
                raw_sparse,
            ],
            [
                args.colmap,
                "model_converter",
                "--input_path",
                raw_sparse / "0",
                "--output_path",
                sparse_dir,
                "--output_type",
                "TXT",
            ],
        ]
    )
    if args.dense:
        commands.extend(
            [
                [
                    args.colmap,
                    "image_undistorter",
                    "--image_path",
                    images_dir,
                    "--input_path",
                    raw_sparse / "0",
                    "--output_path",
                    dense_dir,
                    "--output_type",
                    "COLMAP",
                    "--max_image_size",
                    args.max_image_size,
                ],
                [
                    args.colmap,
                    "patch_match_stereo",
                    "--workspace_path",
                    dense_dir,
                    "--workspace_format",
                    "COLMAP",
                    "--PatchMatchStereo.geom_consistency",
                    "true",
                ],
                [
                    args.colmap,
                    "stereo_fusion",
                    "--workspace_path",
                    dense_dir,
                    "--workspace_format",
                    "COLMAP",
                    "--input_type",
                    "geometric",
                    "--output_path",
                    dense_dir / "fused.ply",
                ],
            ]
        )
    return commands


def write_sparse_seed_ply(sparse_dir: Path, *, dry_run: bool) -> int:
    points_txt = sparse_dir / "points3D.txt"
    points_ply = sparse_dir / "points3D.ply"
    if dry_run:
        return 0
    point_cloud = read_colmap_points3d_txt(points_txt)
    write_binary_ply(points_ply, point_cloud.xyz, point_cloud.rgb)
    return int(point_cloud.xyz.shape[0])


def prepare_dataset(args: argparse.Namespace) -> Path:
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if not args.video.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Osmo 360 source video not found: {args.video}")

    dataset_dir = Path(args.output_root) / args.scene_id
    dirs = reset_output_dirs(args, dataset_dir)
    hfov = args.camera_hfov or fov_from_focal_mm(PRESET_FOCAL_MM[args.perspective_preset])
    commands: list[dict[str, Any]] = []

    run_command(build_perspective_command(args, dirs["images"]), dry_run=args.dry_run, commands=commands)
    for command in colmap_commands(
        args=args,
        images_dir=dirs["images"],
        colmap_dir=dirs["colmap"],
        sparse_dir=dirs["sparse"],
        dense_dir=dirs["dense"],
        hfov_deg=hfov,
    ):
        run_command(command, dry_run=args.dry_run, commands=commands)

    seed_point_count = write_sparse_seed_ply(dirs["sparse"], dry_run=args.dry_run)
    image_count = 0 if args.dry_run else len(list_images(dirs["images"]))
    registered_count = (
        0
        if args.dry_run
        else count_colmap_images_txt(dirs["sparse"] / "images.txt")
    )
    metadata = {
        "dataset": "dji_osmo360",
        "scene_id": args.scene_id,
        "source_video": str(args.video),
        "tool": {
            "name": "Mistral-Yu/360Cam-PGM-3DGS-Tools",
            "path": str(args.tool_root),
            "pinned_commit": PINNED_360CAM_COMMIT,
        },
        "perspective_export": {
            "preset": args.perspective_preset,
            "fps": args.fps,
            "image_size": args.image_size,
            "image_ext": args.image_ext,
            "start": args.start,
            "end": args.end,
            "hfov_deg": hfov,
        },
        "colmap": {
            "matcher": args.matcher,
            "dense": bool(args.dense),
            "max_image_size": args.max_image_size,
            "registered_image_count": registered_count,
            "exported_image_count": image_count,
            "sparse_seed_point_count": seed_point_count,
            "dense_fused_ply": str(dirs["dense"] / "fused.ply") if args.dense else None,
        },
        "commands": commands,
        "dry_run": bool(args.dry_run),
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote prepared Osmo 360 dataset metadata: {dataset_dir / 'metadata.json'}")
    return dataset_dir


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_dir = prepare_dataset(args)
    print(f"Prepared DJI Osmo 360 COLMAP dataset: {dataset_dir}")


if __name__ == "__main__":
    main()
