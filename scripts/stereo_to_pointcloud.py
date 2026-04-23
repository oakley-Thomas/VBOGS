#!/usr/bin/env python3

"""Export a KITTI-360 stereo drive as a world-frame point cloud.

This is the M3 entry point from PLAN.md. It reads rectified perspective stereo
pairs plus cam0 world poses, estimates disparity with a pluggable matcher, and
writes a `points_world.npz` artifact with:

- `xyz`: `(N, 3)` float32 world-frame points
- `rgb`: `(N, 3)` uint8 colors in RGB order
- `frame_id`: `(N,)` int32 source frame ids

An optional `.ply` sidecar can be written for viewer-based sanity checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.data_layout import resolve_kitti360_path


@dataclass(frozen=True)
class StereoCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float


@dataclass(frozen=True)
class FramePose:
    frame_id: int
    c2w: np.ndarray  # (4, 4)


@dataclass(frozen=True)
class StereoFrame:
    frame_id: int
    left_path: Path
    right_path: Path
    pose: FramePose


@dataclass(frozen=True)
class StereoResult:
    disparity_left: np.ndarray
    disparity_right: np.ndarray


class StereoMatcher:
    name = "base"

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> StereoResult:
        raise NotImplementedError


class SGBMStereoMatcher(StereoMatcher):
    name = "sgbm"

    def __init__(
        self,
        *,
        min_disparity: int,
        num_disparities: int,
        block_size: int,
        uniqueness_ratio: int,
        speckle_window_size: int,
        speckle_range: int,
        disp12_max_diff: int,
    ) -> None:
        import cv2

        if num_disparities <= 0 or num_disparities % 16 != 0:
            raise ValueError("--num-disparities must be a positive multiple of 16")
        if block_size <= 0 or block_size % 2 == 0:
            raise ValueError("--block-size must be a positive odd integer")

        def make_matcher(disparity_min: int) -> cv2.StereoSGBM:
            return cv2.StereoSGBM_create(
                minDisparity=disparity_min,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=8 * 3 * block_size * block_size,
                P2=32 * 3 * block_size * block_size,
                disp12MaxDiff=disp12_max_diff,
                uniquenessRatio=uniqueness_ratio,
                speckleWindowSize=speckle_window_size,
                speckleRange=speckle_range,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )

        self.left_matcher = make_matcher(min_disparity)
        # Right-view disparities have the opposite sign. Using a shifted minimum
        # lets us perform a direct left-right consistency check without ximgproc.
        self.right_matcher = make_matcher(-(min_disparity + num_disparities))

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> StereoResult:
        disparity_left = self.left_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disparity_right = self.right_matcher.compute(right_gray, left_gray).astype(np.float32) / 16.0
        return StereoResult(
            disparity_left=disparity_left,
            disparity_right=disparity_right,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive",
        default="2013_05_28_drive_0008_sync",
        help="KITTI-360 drive id to process.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Root containing KITTI-360 rectified stereo images. Defaults to auto-detecting the repo layout.",
    )
    parser.add_argument(
        "--poses-root",
        type=Path,
        default=None,
        help="Root containing KITTI-360 pose text files. Defaults to auto-detecting the repo layout.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Directory containing KITTI-360 calibration text files. Defaults to auto-detecting the repo layout.",
    )
    parser.add_argument(
        "--selection-metadata",
        type=Path,
        default=None,
        help=(
            "Optional metadata.json from `prepare_kitti360_colmap.py`; when set, "
            "uses its `selected_frames` list instead of frame-step sampling."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/points_world"),
        help="Directory where `points_world.npz` and sidecars will be written.",
    )
    parser.add_argument(
        "--output-name",
        default="points_world.npz",
        help="Filename of the exported NPZ artifact.",
    )
    parser.add_argument(
        "--write-ply",
        action="store_true",
        help="Also write a `.ply` copy for viewer-based sanity checks.",
    )
    parser.add_argument(
        "--matcher",
        choices=("sgbm", "raft"),
        default="sgbm",
        help="Stereo provider. `raft` is reserved for a future implementation.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Use every Nth available frame after image/pose matching.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on the number of processed frames.",
    )
    parser.add_argument(
        "--num-disparities",
        type=int,
        default=128,
        help="StereoSGBM search range. Must be divisible by 16.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="StereoSGBM block size.",
    )
    parser.add_argument(
        "--matcher-min-disparity",
        type=int,
        default=0,
        help="Minimum disparity passed to the left-view matcher.",
    )
    parser.add_argument(
        "--min-disparity",
        type=float,
        default=2.0,
        help="Minimum accepted left-view disparity in pixels.",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=80.0,
        help="Maximum retained depth in meters.",
    )
    parser.add_argument(
        "--lr-consistency-threshold",
        type=float,
        default=1.0,
        help="Allowed absolute left-right disparity mismatch in pixels.",
    )
    parser.add_argument(
        "--texture-window-size",
        type=int,
        default=9,
        help="Odd square window used for grayscale texture filtering.",
    )
    parser.add_argument(
        "--texture-threshold",
        type=float,
        default=5.0,
        help="Minimum local grayscale standard deviation for a pixel to survive.",
    )
    parser.add_argument(
        "--pixel-step",
        type=int,
        default=1,
        help="Keep only every Nth valid pixel in x and y to control output size.",
    )
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=250000,
        help="Optional cap on surviving pixels per frame after masking; 0 disables it.",
    )
    parser.add_argument(
        "--uniqueness-ratio",
        type=int,
        default=10,
        help="StereoSGBM uniqueness ratio.",
    )
    parser.add_argument(
        "--speckle-window-size",
        type=int,
        default=50,
        help="StereoSGBM speckle window size.",
    )
    parser.add_argument(
        "--speckle-range",
        type=int,
        default=2,
        help="StereoSGBM speckle range.",
    )
    parser.add_argument(
        "--disp12-max-diff",
        type=int,
        default=1,
        help="StereoSGBM internal left-right disparity check.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used when `--max-points-per-frame` subsamples outputs.",
    )
    return parser.parse_args()


def resolve_input_layout(args: argparse.Namespace) -> None:
    args.raw_root = resolve_kitti360_path(args.raw_root, kind="raw")
    args.poses_root = resolve_kitti360_path(args.poses_root, kind="poses")
    args.calibration_dir = resolve_kitti360_path(args.calibration_dir, kind="calibration")


def parse_perspective_file(path: Path) -> StereoCalibration:
    entries: Dict[str, List[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, raw_values = line.split(":", 1)
            try:
                values = [float(token) for token in raw_values.split()]
            except ValueError:
                continue
            entries[key.strip()] = values

    p0 = np.asarray(entries["P_rect_00"], dtype=np.float64).reshape(3, 4)
    p1 = np.asarray(entries["P_rect_01"], dtype=np.float64).reshape(3, 4)
    size = entries["S_rect_00"]
    fx = float(p0[0, 0])
    fy = float(p0[1, 1])
    cx = float(p0[0, 2])
    cy = float(p0[1, 2])
    baseline = abs(float(p1[0, 3]) / fx)
    return StereoCalibration(
        width=int(size[0]),
        height=int(size[1]),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        baseline_m=baseline,
    )


def parse_cam0_to_world(path: Path) -> Dict[int, FramePose]:
    poses: Dict[int, FramePose] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split()
            if not tokens:
                continue
            frame_id = int(tokens[0])
            values = np.asarray([float(x) for x in tokens[1:]], dtype=np.float64)
            if values.size != 16:
                raise ValueError(
                    f"Expected 16 matrix values for frame {frame_id}, found {values.size}"
                )
            poses[frame_id] = FramePose(frame_id=frame_id, c2w=values.reshape(4, 4))
    if not poses:
        raise ValueError(f"No poses found in {path}")
    return poses


def load_selected_frames(path: Path) -> List[int]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    selected = metadata.get("selected_frames")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"`selected_frames` missing or empty in {path}")
    return [int(frame_id) for frame_id in selected]


def sample_drive_frames(
    left_dir: Path,
    right_dir: Path,
    poses_by_frame: Dict[int, FramePose],
    frame_step: int,
    max_frames: int,
) -> List[StereoFrame]:
    available: List[StereoFrame] = []
    left_images = sorted(left_dir.glob("*.png"))
    for image_path in left_images:
        frame_id = int(image_path.stem)
        right_path = right_dir / image_path.name
        pose = poses_by_frame.get(frame_id)
        if not right_path.exists() or pose is None:
            continue
        available.append(
            StereoFrame(
                frame_id=frame_id,
                left_path=image_path,
                right_path=right_path,
                pose=pose,
            )
        )

    if frame_step <= 0:
        raise ValueError("--frame-step must be positive")

    selected = available[::frame_step]
    if max_frames:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError("No frames selected. Check drive paths, frame step, and poses.")
    return selected


def select_frames_from_metadata(
    left_dir: Path,
    right_dir: Path,
    poses_by_frame: Dict[int, FramePose],
    selected_frame_ids: Sequence[int],
) -> List[StereoFrame]:
    frames: List[StereoFrame] = []
    for frame_id in selected_frame_ids:
        image_name = f"{frame_id:010d}.png"
        left_path = left_dir / image_name
        right_path = right_dir / image_name
        pose = poses_by_frame.get(frame_id)
        if not left_path.exists():
            raise FileNotFoundError(f"Selected left image not found: {left_path}")
        if not right_path.exists():
            raise FileNotFoundError(f"Selected right image not found: {right_path}")
        if pose is None:
            raise KeyError(f"Selected frame {frame_id} is missing from the pose file")
        frames.append(
            StereoFrame(
                frame_id=frame_id,
                left_path=left_path,
                right_path=right_path,
                pose=pose,
            )
        )
    if not frames:
        raise ValueError("Selected frame list is empty.")
    return frames


def build_matcher(args: argparse.Namespace) -> StereoMatcher:
    if args.matcher == "sgbm":
        return SGBMStereoMatcher(
            min_disparity=args.matcher_min_disparity,
            num_disparities=args.num_disparities,
            block_size=args.block_size,
            uniqueness_ratio=args.uniqueness_ratio,
            speckle_window_size=args.speckle_window_size,
            speckle_range=args.speckle_range,
            disp12_max_diff=args.disp12_max_diff,
        )
    if args.matcher == "raft":
        raise NotImplementedError(
            "`--matcher raft` is reserved for a future provider implementation. "
            "The CLI and output contract are in place; use `sgbm` today."
        )
    raise ValueError(f"Unsupported matcher: {args.matcher}")


def compute_texture_mask(gray: np.ndarray, window_size: int, threshold: float) -> np.ndarray:
    import cv2

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("--texture-window-size must be a positive odd integer")

    gray_f32 = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f32, ddepth=-1, ksize=(window_size, window_size), normalize=True)
    sq_mean = cv2.boxFilter(
        gray_f32 * gray_f32,
        ddepth=-1,
        ksize=(window_size, window_size),
        normalize=True,
    )
    variance = np.maximum(sq_mean - mean * mean, 0.0)
    return np.sqrt(variance) >= threshold


def build_validity_mask(
    disparity_left: np.ndarray,
    disparity_right: np.ndarray,
    left_gray: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    height, width = disparity_left.shape
    xs = np.arange(width, dtype=np.float32)[None, :].repeat(height, axis=0)
    x_right = xs - disparity_left

    in_bounds = np.isfinite(x_right) & (x_right >= 0.0) & (x_right <= (width - 1))
    right_indices = np.clip(np.rint(x_right).astype(np.int32), 0, width - 1)
    sampled_right_disp = disparity_right[np.arange(height)[:, None], right_indices]
    lr_consistent = np.abs(disparity_left + sampled_right_disp) <= args.lr_consistency_threshold

    disparity_valid = np.isfinite(disparity_left) & (disparity_left > args.min_disparity)
    texture_valid = compute_texture_mask(
        left_gray,
        window_size=args.texture_window_size,
        threshold=args.texture_threshold,
    )

    valid = disparity_valid & in_bounds & np.isfinite(sampled_right_disp) & lr_consistent & texture_valid

    if args.pixel_step > 1:
        ys = np.arange(height)[:, None]
        xs_idx = np.arange(width)[None, :]
        valid &= (ys % args.pixel_step == 0) & (xs_idx % args.pixel_step == 0)

    return valid


def unproject_to_world(
    *,
    disparity_left: np.ndarray,
    rgb_image: np.ndarray,
    calibration: StereoCalibration,
    c2w: np.ndarray,
    valid_mask: np.ndarray,
    max_points_per_frame: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(valid_mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    disparities = disparity_left[ys, xs]
    depth = (calibration.fx * calibration.baseline_m) / disparities
    depth_valid = np.isfinite(depth) & (depth > 0.0)
    ys = ys[depth_valid]
    xs = xs[depth_valid]
    depth = depth[depth_valid]
    if depth.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    if max_points_per_frame > 0 and depth.size > max_points_per_frame:
        keep = rng.choice(depth.size, size=max_points_per_frame, replace=False)
        ys = ys[keep]
        xs = xs[keep]
        depth = depth[keep]

    x_cam = (xs.astype(np.float64) - calibration.cx) * depth / calibration.fx
    y_cam = (ys.astype(np.float64) - calibration.cy) * depth / calibration.fy
    z_cam = depth.astype(np.float64)
    points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)
    points_world = (c2w @ points_cam.T).T[:, :3].astype(np.float32)
    colors = rgb_image[ys, xs].astype(np.uint8)
    return points_world, colors


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    normals = np.zeros_like(xyz, dtype=np.float32)
    vertex_data = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    attributes = np.concatenate([xyz, normals, rgb], axis=1)
    vertex_data[:] = list(map(tuple, attributes))
    ply = PlyData([PlyElement.describe(vertex_data, "vertex")], text=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    ply.write(path)


def export_points(args: argparse.Namespace) -> Path:
    import cv2

    drive_root = args.raw_root / args.drive
    left_dir = drive_root / "image_00" / "data_rect"
    right_dir = drive_root / "image_01" / "data_rect"
    poses_path = args.poses_root / args.drive / "cam0_to_world.txt"
    perspective_path = args.calibration_dir / "perspective.txt"

    if not left_dir.exists():
        raise FileNotFoundError(f"Left image directory not found: {left_dir}")
    if not right_dir.exists():
        raise FileNotFoundError(f"Right image directory not found: {right_dir}")
    if not poses_path.exists():
        raise FileNotFoundError(f"Pose file not found: {poses_path}")
    if not perspective_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {perspective_path}")

    calibration = parse_perspective_file(perspective_path)
    poses_by_frame = parse_cam0_to_world(poses_path)
    if args.selection_metadata is not None:
        selected_frame_ids = load_selected_frames(args.selection_metadata)
        frames = select_frames_from_metadata(left_dir, right_dir, poses_by_frame, selected_frame_ids)
    else:
        frames = sample_drive_frames(left_dir, right_dir, poses_by_frame, args.frame_step, args.max_frames)

    output_dir = args.output_root / args.drive
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    metadata_path = output_dir / f"{output_path.stem}_metadata.json"
    ply_path = output_dir / f"{output_path.stem}.ply"

    matcher = build_matcher(args)
    rng = np.random.default_rng(args.random_seed)

    all_xyz: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []
    all_frame_ids: List[np.ndarray] = []
    frame_stats: List[dict] = []

    for frame in frames:
        left = cv2.imread(str(frame.left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(frame.right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise RuntimeError(f"Failed to read stereo pair for frame {frame.frame_id}")

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        rgb_left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

        stereo = matcher.compute(left_gray, right_gray)
        valid_mask = build_validity_mask(stereo.disparity_left, stereo.disparity_right, left_gray, args)
        points_world, colors = unproject_to_world(
            disparity_left=stereo.disparity_left,
            rgb_image=rgb_left,
            calibration=calibration,
            c2w=frame.pose.c2w,
            valid_mask=valid_mask,
            max_points_per_frame=args.max_points_per_frame,
            rng=rng,
        )

        frame_stats.append(
            {
                "frame_id": frame.frame_id,
                "valid_pixels": int(valid_mask.sum()),
                "points_kept": int(points_world.shape[0]),
            }
        )
        print(
            f"[stereo] frame {frame.frame_id:010d}: "
            f"valid_pixels={int(valid_mask.sum())} kept={points_world.shape[0]}"
        )

        if points_world.size == 0:
            continue

        all_xyz.append(points_world)
        all_rgb.append(colors)
        all_frame_ids.append(
            np.full(points_world.shape[0], frame.frame_id, dtype=np.int32)
        )

    if not all_xyz:
        raise RuntimeError("Stereo export produced no valid world points.")

    xyz = np.concatenate(all_xyz, axis=0).astype(np.float32, copy=False)
    rgb = np.concatenate(all_rgb, axis=0).astype(np.uint8, copy=False)
    frame_id = np.concatenate(all_frame_ids, axis=0).astype(np.int32, copy=False)

    np.savez_compressed(output_path, xyz=xyz, rgb=rgb, frame_id=frame_id)
    if args.write_ply:
        write_ply(ply_path, xyz, rgb)

    metadata = {
        "drive": args.drive,
        "matcher": matcher.name,
        "output_npz": str(output_path),
        "output_ply": str(ply_path) if args.write_ply else None,
        "num_frames": len(frames),
        "num_points": int(xyz.shape[0]),
        "selection_metadata": str(args.selection_metadata) if args.selection_metadata else None,
        "frame_step": args.frame_step,
        "max_frames": args.max_frames,
        "intrinsics": {
            "width": calibration.width,
            "height": calibration.height,
            "fx": calibration.fx,
            "fy": calibration.fy,
            "cx": calibration.cx,
            "cy": calibration.cy,
            "baseline_m": calibration.baseline_m,
        },
        "filters": {
            "min_disparity": args.min_disparity,
            "max_depth_m": args.max_depth_m,
            "lr_consistency_threshold": args.lr_consistency_threshold,
            "texture_window_size": args.texture_window_size,
            "texture_threshold": args.texture_threshold,
            "pixel_step": args.pixel_step,
            "max_points_per_frame": args.max_points_per_frame,
        },
        "frame_stats": frame_stats,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    resolve_input_layout(args)
    output_path = export_points(args)
    print(f"Wrote world point cloud to: {output_path}")


if __name__ == "__main__":
    main()
