#!/usr/bin/env python3

"""Serve a realtime browser viewer for a trained Octree-AnyGS scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.viewer.rendering import DEFAULT_DRIVE, DEFAULT_VIEWER_PORT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", default=DEFAULT_DRIVE)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Octree-AnyGS run directory. Defaults to the latest run for --drive.",
    )
    parser.add_argument(
        "--u-path",
        type=Path,
        default=None,
        help="Per-anchor uncertainty array. Defaults to `data/m4/<drive>/U.npy`.",
    )
    parser.add_argument("--octree-root", type=Path, default=Path("Octree-AnyGS"))
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument(
        "--resolution",
        type=int,
        default=4,
        help="Octree-AnyGS image divisor/target width for interactive rendering.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_VIEWER_PORT)
    parser.add_argument("--max-fps", type=float, default=20.0)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--camera-source", choices=("train", "test"), default="test")
    parser.add_argument("--camera-index", type=int, default=0)
    pose_group = parser.add_mutually_exclusive_group()
    pose_group.add_argument(
        "--initial-pose",
        nargs="+",
        default=None,
        help=(
            "Initial viewer pose as `x y z yaw pitch roll` in degrees, "
            "or 12/16 row-major matrix values."
        ),
    )
    pose_group.add_argument(
        "--initial-pose-file",
        type=Path,
        default=None,
        help="Text, JSON, .npy, or .npz file containing the initial viewer pose.",
    )
    parser.add_argument(
        "--initial-pose-convention",
        choices=("c2w", "w2c"),
        default="c2w",
        help="Convention for --initial-pose matrix input or generic matrix files.",
    )
    parser.add_argument("--colormap", default="turbo")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--force-all-levels", action="store_true")
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="Start without U.npy and expose only the RGB layer.",
    )
    parser.add_argument(
        "--load-source-images",
        action="store_true",
        help=(
            "Use the original Octree-AnyGS camera loader, including source image "
            "tensors. The default viewer path skips those tensors to reduce RAM/VRAM."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "The realtime viewer requires `uvicorn`. Rebuild vbogs-torch or install `fastapi uvicorn[standard]`."
        ) from exc

    from vbogs.viewer.rendering import OctreeRenderSession
    from vbogs.viewer.pose import load_pose_file, parse_pose_to_c2w, pose_to_c2w
    from vbogs.viewer.server import create_app

    initial_c2w = None
    if args.initial_pose is not None:
        initial_c2w = parse_pose_to_c2w(args.initial_pose, convention=args.initial_pose_convention)
    elif args.initial_pose_file is not None:
        pose_matrix, convention = load_pose_file(args.initial_pose_file, args.initial_pose_convention)
        initial_c2w = pose_to_c2w(pose_matrix, convention)

    session = OctreeRenderSession(
        drive=args.drive,
        model_path=args.model_path,
        u_path=args.u_path,
        octree_root=args.octree_root,
        iteration=args.iteration,
        resolution=args.resolution,
        camera_source=args.camera_source,
        camera_index=args.camera_index,
        colormap=args.colormap,
        vmin=args.vmin,
        vmax=args.vmax,
        force_all_levels=args.force_all_levels,
        rgb_only=args.rgb_only,
        load_source_images=args.load_source_images,
        jpeg_quality=args.jpeg_quality,
        max_fps=args.max_fps,
        initial_c2w=initial_c2w,
    )
    app = create_app(session)
    print(f"Serving VBOGS viewer at http://{args.host}:{args.port}")
    # Octree-AnyGS `safe_state()` wraps stdout with a timestamping object that
    # does not implement every file-like method Uvicorn's default logging
    # formatter expects. Keep the server logs simple after scene initialization.
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
