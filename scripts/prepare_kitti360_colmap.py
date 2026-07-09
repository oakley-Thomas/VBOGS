#!/usr/bin/env python3

"""Prepare a KITTI-360 drive in Octree-AnyGS's COLMAP-style layout.

This script adapts the local KITTI-360 perspective stereo layout into the
`images/` + `sparse/0/` structure that Octree-AnyGS expects for `data_format:
colmap`. It writes:

- `images/*.png` as symlinks or copies to the rectified left-camera images by default
- `sparse/0/cameras.txt`
- `sparse/0/images.txt`
- `sparse/0/points3D.ply`
- `metadata.json` describing the conversion inputs

The sparse point cloud is bootstrapped from lightweight stereo depth rather
than COLMAP, which keeps M2 self-contained and runnable on the dev machine.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.data_layout import resolve_kitti360_path

LIDAR_FALLBACK_GRAY = 160


@dataclass(frozen=True)
class PinholeCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class StereoCalibration:
    left: PinholeCalibration
    right: PinholeCalibration
    right_center_in_left: np.ndarray
    baseline_m: float
    r_rect_00: np.ndarray | None = None

    @property
    def width(self) -> int:
        return self.left.width

    @property
    def height(self) -> int:
        return self.left.height

    @property
    def fx(self) -> float:
        return self.left.fx

    @property
    def fy(self) -> float:
        return self.left.fy

    @property
    def cx(self) -> float:
        return self.left.cx

    @property
    def cy(self) -> float:
        return self.left.cy


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    label: str
    calibration: PinholeCalibration


@dataclass(frozen=True)
class FramePose:
    frame_id: int
    c2w: np.ndarray  # (4, 4)


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    camera_id: int
    image_name: str
    c2w: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive",
        default="2013_05_28_drive_0008_sync",
        help="KITTI-360 drive id to prepare.",
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
        "--output-root",
        type=Path,
        default=Path("/data/COLMAP"),
        help="Root directory for prepared Octree-AnyGS datasets.",
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
        help="Optional cap on the number of frames copied into the dataset.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Whether to symlink or copy image files into the dataset.",
    )
    parser.add_argument(
        "--training-cameras",
        choices=("left", "stereo"),
        default="left",
        help=(
            "RGB cameras written for Octree-AnyGS training. `left` preserves the "
            "legacy image_00-only layout; `stereo` adds image_01 as a second "
            "posed camera."
        ),
    )
    parser.add_argument(
        "--seed-mode",
        choices=("stereo", "lidar", "random"),
        default="lidar",
        help=(
            "How to bootstrap the sparse seed point cloud. `lidar` (default) uses "
            "raw velodyne scans; `stereo` uses SGBM depth; `random` samples a box "
            "around the camera path."
        ),
    )
    parser.add_argument(
        "--velodyne-root",
        type=Path,
        default=None,
        help=(
            "Root containing KITTI-360 raw velodyne scans (data_3d_raw). Defaults "
            "to auto-detecting the repo layout. Used by `--seed-mode lidar`."
        ),
    )
    parser.add_argument(
        "--seed-max-points",
        type=int,
        default=None,
        help=(
            "Cap on total seed points written to points3D.ply. Defaults to "
            "--stereo-max-points for backward compatibility."
        ),
    )
    parser.add_argument(
        "--max-points-per-lidar-frame",
        type=int,
        default=5000,
        help="Cap on sampled lidar points per frame before the global cap.",
    )
    parser.add_argument(
        "--lidar-min-range-m",
        type=float,
        default=2.5,
        help="Minimum radial range kept from each velodyne scan (drops ego returns).",
    )
    parser.add_argument(
        "--lidar-max-range-m",
        type=float,
        default=80.0,
        help="Maximum radial range kept from each velodyne scan.",
    )
    parser.add_argument(
        "--stereo-max-points",
        type=int,
        default=60000,
        help="Cap on total sparse bootstrap points written to points3D.ply.",
    )
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=1500,
        help="Cap on sampled stereo points per frame before the global cap.",
    )
    parser.add_argument(
        "--stereo-pixel-step",
        type=int,
        default=8,
        help="Stride used when subsampling valid disparity pixels.",
    )
    parser.add_argument(
        "--num-disparities",
        type=int,
        default=128,
        help="StereoSGBM num disparities. Must be divisible by 16.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="StereoSGBM block size.",
    )
    parser.add_argument(
        "--min-disparity",
        type=float,
        default=2.0,
        help="Minimum valid disparity in pixels.",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=80.0,
        help="Maximum depth retained from the sparse stereo bootstrap.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used for point subsampling and random fallback.",
    )
    return parser.parse_args()


def resolve_input_layout(args: argparse.Namespace) -> None:
    args.raw_root = resolve_kitti360_path(args.raw_root, kind="raw")
    args.poses_root = resolve_kitti360_path(args.poses_root, kind="poses")
    args.calibration_dir = resolve_kitti360_path(args.calibration_dir, kind="calibration")
    args.velodyne_root = resolve_kitti360_path(args.velodyne_root, kind="velodyne")


def camera_from_projection(projection: np.ndarray, size: Sequence[float]) -> PinholeCalibration:
    return PinholeCalibration(
        width=int(size[0]),
        height=int(size[1]),
        fx=float(projection[0, 0]),
        fy=float(projection[1, 1]),
        cx=float(projection[0, 2]),
        cy=float(projection[1, 2]),
    )


def camera_center_from_projection(projection: np.ndarray) -> np.ndarray:
    matrix = np.asarray(projection[:, :3], dtype=np.float64)
    offset = np.asarray(projection[:, 3], dtype=np.float64)
    return -np.linalg.solve(matrix, offset)


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
    left_size = entries["S_rect_00"]
    right_size = entries.get("S_rect_01", left_size)
    left = camera_from_projection(p0, left_size)
    right = camera_from_projection(p1, right_size)
    left_center = camera_center_from_projection(p0)
    right_center = camera_center_from_projection(p1)
    right_center_in_left = right_center - left_center
    baseline = abs(float(right_center_in_left[0]))
    r_rect_values = entries.get("R_rect_00")
    r_rect_00 = (
        np.asarray(r_rect_values, dtype=np.float64).reshape(3, 3)
        if r_rect_values is not None and len(r_rect_values) == 9
        else None
    )
    return StereoCalibration(
        left=left,
        right=right,
        right_center_in_left=right_center_in_left.astype(np.float64),
        baseline_m=baseline,
        r_rect_00=r_rect_00,
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
            c2w = values.reshape(4, 4)
            poses[frame_id] = FramePose(frame_id=frame_id, c2w=c2w)
    if not poses:
        raise ValueError(f"No poses found in {path}")
    return poses


def parse_cam_to_velo(path: Path) -> np.ndarray:
    """Parse calib_cam_to_velo.txt (12 floats, 3x4) into a 4x4 cam0->velo transform."""

    values = np.asarray(
        [float(token) for token in path.read_text(encoding="utf-8").split()],
        dtype=np.float64,
    )
    if values.size != 12:
        raise ValueError(f"Expected 12 values in {path}, found {values.size}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :4] = values.reshape(3, 4)
    return transform


def load_velodyne_scan(path: Path) -> np.ndarray:
    """Load a raw KITTI-360 velodyne scan as (N, 4) float32 x, y, z, intensity."""

    scan = np.fromfile(path, dtype=np.float32)
    if scan.size % 4 != 0:
        raise ValueError(f"Velodyne scan size {scan.size} is not divisible by 4: {path}")
    return scan.reshape(-1, 4)


def rotmat_to_qvec(rot: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rot.flat
    k = np.array(
        [
            [rxx - ryy - rzz, 0.0, 0.0, 0.0],
            [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def materialize_image(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def translation_matrix(offset: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(offset, dtype=np.float64).reshape(3)
    return transform


def right_camera_c2w(left_c2w: np.ndarray, calibration: StereoCalibration) -> np.ndarray:
    return np.asarray(left_c2w, dtype=np.float64).reshape(4, 4) @ translation_matrix(
        calibration.right_center_in_left
    )


def build_matcher(args: argparse.Namespace) -> cv2.StereoSGBM:
    import cv2

    if args.num_disparities % 16 != 0:
        raise ValueError("--num-disparities must be divisible by 16")
    block = args.block_size
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=args.num_disparities,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def sample_drive_frames(
    left_dir: Path,
    right_dir: Path,
    poses_by_frame: Dict[int, FramePose],
    frame_step: int,
    max_frames: int,
) -> List[Tuple[int, Path, Path, FramePose]]:
    available: List[Tuple[int, Path, Path, FramePose]] = []
    left_images = sorted(left_dir.glob("*.png"))
    for image_path in left_images:
        frame_id = int(image_path.stem)
        right_path = right_dir / image_path.name
        if not right_path.exists():
            continue
        pose = poses_by_frame.get(frame_id)
        if pose is None:
            continue
        available.append((frame_id, image_path, right_path, pose))

    if frame_step <= 0:
        raise ValueError("--frame-step must be positive")

    selected = available[::frame_step]
    if max_frames:
        selected = selected[:max_frames]

    if not selected:
        raise ValueError("No frames selected. Check drive paths, frame step, and poses.")

    return selected


def image_to_world_points(
    disparity: np.ndarray,
    rgb_image: np.ndarray,
    calibration: StereoCalibration,
    c2w: np.ndarray,
    pixel_step: int,
    min_disparity: float,
    max_depth_m: float,
    max_points_per_frame: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(disparity) & (disparity > min_disparity)
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    stride_mask = ((ys % pixel_step) == 0) & ((xs % pixel_step) == 0)
    ys = ys[stride_mask]
    xs = xs[stride_mask]
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    disp_values = disparity[ys, xs]
    depth = (calibration.fx * calibration.baseline_m) / disp_values
    depth_mask = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth_m)
    ys = ys[depth_mask]
    xs = xs[depth_mask]
    depth = depth[depth_mask]
    if depth.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    if depth.size > max_points_per_frame:
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


def build_sparse_points_from_stereo(
    frames: Sequence[Tuple[int, Path, Path, FramePose]],
    calibration: StereoCalibration,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    import cv2

    matcher = build_matcher(args)
    rng = np.random.default_rng(args.random_seed)
    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []

    for frame_id, left_path, right_path, pose in frames:
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            continue

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        rgb_left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        points_world, colors = image_to_world_points(
            disparity=disparity,
            rgb_image=rgb_left,
            calibration=calibration,
            c2w=pose.c2w,
            pixel_step=args.stereo_pixel_step,
            min_disparity=args.min_disparity,
            max_depth_m=args.max_depth_m,
            max_points_per_frame=args.max_points_per_frame,
            rng=rng,
        )
        if points_world.size == 0:
            continue
        all_points.append(points_world)
        all_colors.append(colors)
        print(f"[stereo] frame {frame_id:010d}: kept {points_world.shape[0]} sparse points")

    if not all_points:
        raise RuntimeError("Stereo bootstrap produced no valid sparse points.")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    max_points = seed_point_cap(args)
    if points.shape[0] > max_points:
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    seed_metadata = {
        "seed_source": "stereo_sgbm",
        "seed_point_count": int(points.shape[0]),
    }
    return points, colors, seed_metadata


def seed_point_cap(args: argparse.Namespace) -> int:
    if args.seed_max_points is not None:
        return args.seed_max_points
    return args.stereo_max_points


def sample_lidar_point_colors(
    points_cam: np.ndarray,
    image_path: Path,
    calibration: PinholeCalibration,
) -> Tuple[np.ndarray, int]:
    """Sample RGB for rectified-cam0 points by projecting into the left image.

    Points behind the camera, out of view, or without a readable image keep the
    flat gray fallback color. Returns the colors and how many points were colored.
    """

    colors = np.full((points_cam.shape[0], 3), LIDAR_FALLBACK_GRAY, dtype=np.uint8)
    try:
        import cv2
    except ImportError:
        return colors, 0
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return colors, 0
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    z = points_cam[:, 2]
    in_front = z > 0.1
    us = np.full(z.shape, -1, dtype=np.int64)
    vs = np.full(z.shape, -1, dtype=np.int64)
    us[in_front] = np.round(
        points_cam[in_front, 0] / z[in_front] * calibration.fx + calibration.cx
    ).astype(np.int64)
    vs[in_front] = np.round(
        points_cam[in_front, 1] / z[in_front] * calibration.fy + calibration.cy
    ).astype(np.int64)
    visible = (
        in_front
        & (us >= 0)
        & (us < rgb.shape[1])
        & (vs >= 0)
        & (vs < rgb.shape[0])
    )
    colors[visible] = rgb[vs[visible], us[visible]]
    return colors, int(np.count_nonzero(visible))


def build_sparse_points_from_lidar(
    frames: Sequence[Tuple[int, Path, Path, FramePose]],
    calibration: StereoCalibration,
    velodyne_dir: Path,
    velo_to_cam0_rect: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(args.random_seed)
    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []
    seed_frames: List[int] = []
    frames_missing = 0
    colored_points = 0
    total_points = 0

    for frame_id, left_path, _right_path, pose in frames:
        scan_path = velodyne_dir / f"{frame_id:010d}.bin"
        if not scan_path.exists():
            frames_missing += 1
            continue
        scan = load_velodyne_scan(scan_path)
        ranges = np.linalg.norm(scan[:, :3], axis=1)
        in_range = (ranges >= args.lidar_min_range_m) & (ranges <= args.lidar_max_range_m)
        points_velo = scan[in_range, :3].astype(np.float64)
        if points_velo.shape[0] == 0:
            continue
        if points_velo.shape[0] > args.max_points_per_lidar_frame:
            sample = rng.choice(
                points_velo.shape[0], size=args.max_points_per_lidar_frame, replace=False
            )
            points_velo = points_velo[sample]

        homogeneous = np.concatenate(
            [points_velo, np.ones((points_velo.shape[0], 1))], axis=1
        )
        points_cam = (velo_to_cam0_rect @ homogeneous.T).T[:, :3]
        colors, frame_colored = sample_lidar_point_colors(
            points_cam, left_path, calibration.left
        )
        points_cam_h = np.concatenate(
            [points_cam, np.ones((points_cam.shape[0], 1))], axis=1
        )
        points_world = (pose.c2w @ points_cam_h.T).T[:, :3].astype(np.float32)

        all_points.append(points_world)
        all_colors.append(colors)
        seed_frames.append(frame_id)
        colored_points += frame_colored
        total_points += points_world.shape[0]
        print(f"[lidar] frame {frame_id:010d}: kept {points_world.shape[0]} seed points")

    if not all_points:
        raise RuntimeError(
            f"LiDAR bootstrap found no velodyne scans under {velodyne_dir}. "
            "Download the drive's data_3d_raw archive or rerun with --seed-mode stereo."
        )

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    max_points = seed_point_cap(args)
    if points.shape[0] > max_points:
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    seed_metadata = {
        "seed_source": "velodyne",
        "seed_frames": seed_frames,
        "seed_point_count": int(points.shape[0]),
        "frames_missing_velodyne": frames_missing,
        "colored_point_fraction": float(colored_points / total_points),
        "lidar_min_range_m": float(args.lidar_min_range_m),
        "lidar_max_range_m": float(args.lidar_max_range_m),
    }
    return points, colors, seed_metadata


def build_random_points(
    frames: Sequence[Tuple[int, Path, Path, FramePose]],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(args.random_seed)
    camera_centers = np.stack([frame_pose.c2w[:3, 3] for _, _, _, frame_pose in frames], axis=0)
    center_min = camera_centers.min(axis=0)
    center_max = camera_centers.max(axis=0)
    span = np.maximum(center_max - center_min, 1.0)
    num_points = min(seed_point_cap(args), max(5000, len(frames) * 256))
    points = rng.uniform(center_min - 0.5 * span, center_max + 0.5 * span, size=(num_points, 3))
    colors = rng.integers(0, 255, size=(num_points, 3), dtype=np.uint8)
    seed_metadata = {
        "seed_source": "random",
        "seed_point_count": int(num_points),
    }
    return points.astype(np.float32), colors, seed_metadata


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


def write_cameras_txt(path: Path, cameras: Sequence[ColmapCamera]) -> None:
    header = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    lines = header.copy()
    for camera in cameras:
        calibration = camera.calibration
        lines.append(
            f"{camera.camera_id} PINHOLE {calibration.width} {calibration.height} "
            f"{calibration.fx:.8f} {calibration.fy:.8f} "
            f"{calibration.cx:.8f} {calibration.cy:.8f}"
        )
    path.write_text("\n".join(lines + [""]) + "\n", encoding="utf-8")


def write_images_txt(path: Path, images: Sequence[ColmapImage]) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(images)}",
    ]
    for image in images:
        c2w = np.asarray(image.c2w, dtype=np.float64).reshape(4, 4)
        r_wc = c2w[:3, :3]
        t_wc = c2w[:3, 3]
        r_cw = r_wc.T
        t_cw = -r_cw @ t_wc
        qvec = rotmat_to_qvec(r_cw)
        lines.append(
            f"{image.image_id} "
            f"{qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
            f"{t_cw[0]:.12f} {t_cw[1]:.12f} {t_cw[2]:.12f} "
            f"{image.camera_id} {image.image_name}"
        )
        # One dummy feature observation is enough for the text parser path.
        lines.append("0.0 0.0 -1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_training_cameras(
    calibration: StereoCalibration,
    training_cameras: str,
) -> list[ColmapCamera]:
    cameras = [
        ColmapCamera(camera_id=1, label="image_00", calibration=calibration.left)
    ]
    if training_cameras == "stereo":
        cameras.append(
            ColmapCamera(camera_id=2, label="image_01", calibration=calibration.right)
        )
    return cameras


def matrix_for_metadata(matrix: np.ndarray) -> list[list[float]]:
    return [
        [float(value) for value in row]
        for row in np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    ]


def calibration_for_metadata(
    calibration: PinholeCalibration,
    camera_id: int,
) -> dict[str, float | int]:
    return {
        "camera_id": int(camera_id),
        "width": int(calibration.width),
        "height": int(calibration.height),
        "fx": float(calibration.fx),
        "fy": float(calibration.fy),
        "cx": float(calibration.cx),
        "cy": float(calibration.cy),
    }


def materialize_training_images(
    *,
    frames: Sequence[Tuple[int, Path, Path, FramePose]],
    calibration: StereoCalibration,
    training_cameras: str,
    images_out: Path,
    copy_mode: str,
) -> tuple[list[ColmapImage], list[dict]]:
    colmap_images: list[ColmapImage] = []
    frame_records: list[dict] = []
    image_id = 1

    for frame_id, left_path, right_path, pose in frames:
        output_name = f"{frame_id:010d}.png"
        frame_images: list[dict] = []

        camera_specs: list[tuple[str, int, Path, str, np.ndarray]]
        if training_cameras == "stereo":
            camera_specs = [
                ("image_00", 1, left_path, f"image_00/{output_name}", pose.c2w),
                (
                    "image_01",
                    2,
                    right_path,
                    f"image_01/{output_name}",
                    right_camera_c2w(pose.c2w, calibration),
                ),
            ]
        else:
            camera_specs = [("image_00", 1, left_path, output_name, pose.c2w)]

        for camera_label, camera_id, src_path, image_name, c2w in camera_specs:
            materialize_image(src_path, images_out / image_name, copy_mode)
            colmap_images.append(
                ColmapImage(
                    image_id=image_id,
                    camera_id=camera_id,
                    image_name=image_name,
                    c2w=c2w,
                )
            )
            frame_images.append(
                {
                    "camera": camera_label,
                    "camera_id": camera_id,
                    "image_name": image_name,
                    "source_path": str(src_path),
                    "c2w": matrix_for_metadata(c2w),
                }
            )
            image_id += 1

        frame_records.append({"frame_id": frame_id, "images": frame_images})

    return colmap_images, frame_records


def prepare_dataset(args: argparse.Namespace) -> Path:
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
    frames = sample_drive_frames(
        left_dir=left_dir,
        right_dir=right_dir,
        poses_by_frame=poses_by_frame,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )

    dataset_dir = args.output_root / args.drive
    images_out = dataset_dir / "images"
    sparse_out = dataset_dir / "sparse" / "0"
    ensure_empty_dir(dataset_dir)
    images_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    training_cameras = getattr(args, "training_cameras", "left")
    cameras = selected_training_cameras(calibration, training_cameras)
    colmap_images, frame_records = materialize_training_images(
        frames=frames,
        calibration=calibration,
        training_cameras=training_cameras,
        images_out=images_out,
        copy_mode=args.copy_mode,
    )

    if args.seed_mode == "stereo":
        points_xyz, points_rgb, seed_metadata = build_sparse_points_from_stereo(
            frames, calibration, args
        )
    elif args.seed_mode == "lidar":
        cam_to_velo_path = args.calibration_dir / "calib_cam_to_velo.txt"
        if not cam_to_velo_path.exists():
            raise FileNotFoundError(
                f"LiDAR seeding requires {cam_to_velo_path}. Download the KITTI-360 "
                "calibration archive or rerun with --seed-mode stereo."
            )
        if calibration.r_rect_00 is None:
            raise ValueError(
                f"LiDAR seeding requires R_rect_00 in {perspective_path}. "
                "Rerun with --seed-mode stereo if it is unavailable."
            )
        velodyne_dir = args.velodyne_root / args.drive / "velodyne_points" / "data"
        if not velodyne_dir.exists():
            raise FileNotFoundError(
                f"Velodyne scan directory not found: {velodyne_dir}. Download the "
                "drive's data_3d_raw archive or rerun with --seed-mode stereo."
            )
        cam0_to_velo = parse_cam_to_velo(cam_to_velo_path)
        r_rect = np.eye(4, dtype=np.float64)
        r_rect[:3, :3] = calibration.r_rect_00
        velo_to_cam0_rect = r_rect @ np.linalg.inv(cam0_to_velo)
        points_xyz, points_rgb, seed_metadata = build_sparse_points_from_lidar(
            frames=frames,
            calibration=calibration,
            velodyne_dir=velodyne_dir,
            velo_to_cam0_rect=velo_to_cam0_rect,
            args=args,
        )
    else:
        points_xyz, points_rgb, seed_metadata = build_random_points(frames, args)

    write_cameras_txt(sparse_out / "cameras.txt", cameras)
    write_images_txt(sparse_out / "images.txt", colmap_images)
    write_ply(sparse_out / "points3D.ply", points_xyz, points_rgb)

    camera_intrinsics = {
        camera.label: calibration_for_metadata(camera.calibration, camera.camera_id)
        for camera in cameras
    }
    metadata = {
        "drive": args.drive,
        "num_frames": len(frames),
        "num_images": len(colmap_images),
        "training_cameras": training_cameras,
        "frame_step": args.frame_step,
        "max_frames": args.max_frames,
        "copy_mode": args.copy_mode,
        "seed_mode": args.seed_mode,
        "seed_metadata": seed_metadata,
        "stereo_max_points": int(points_xyz.shape[0]),
        "camera_intrinsics": camera_intrinsics,
        "right_camera_center_in_left": [
            float(value) for value in calibration.right_center_in_left
        ],
        "intrinsics": {
            "width": calibration.width,
            "height": calibration.height,
            "fx": calibration.fx,
            "fy": calibration.fy,
            "cx": calibration.cx,
            "cy": calibration.cy,
            "baseline_m": calibration.baseline_m,
        },
        "selected_frames": [frame_id for frame_id, _, _, _ in frames],
        "frame_records": frame_records,
    }
    if args.seed_mode == "lidar":
        metadata["velodyne_root"] = str(args.velodyne_root)
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return dataset_dir


def main() -> None:
    args = parse_args()
    resolve_input_layout(args)
    dataset_dir = prepare_dataset(args)
    print(f"Prepared Octree-AnyGS dataset at: {dataset_dir}")


if __name__ == "__main__":
    main()
