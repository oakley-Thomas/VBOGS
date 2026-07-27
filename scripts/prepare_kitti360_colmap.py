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
from vbogs.dataset_splits import split_frame_indices, split_lookup
from vbogs.dynamic_masking import load_manifest, mask_path, read_static_mask


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
        choices=("stereo", "random"),
        default="stereo",
        help="How to bootstrap the sparse seed point cloud.",
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
    parser.add_argument(
        "--dynamic-mask-root",
        type=Path,
        default=None,
        help="Optional dynamic-mask artifact. Its masks are mirrored into the prepared dataset.",
    )
    return parser.parse_args()


def resolve_input_layout(args: argparse.Namespace) -> None:
    args.raw_root = resolve_kitti360_path(args.raw_root, kind="raw")
    args.poses_root = resolve_kitti360_path(args.poses_root, kind="poses")
    args.calibration_dir = resolve_kitti360_path(args.calibration_dir, kind="calibration")


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
    return StereoCalibration(
        left=left,
        right=right,
        right_center_in_left=right_center_in_left.astype(np.float64),
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
            c2w = values.reshape(4, 4)
            poses[frame_id] = FramePose(frame_id=frame_id, c2w=c2w)
    if not poses:
        raise ValueError(f"No poses found in {path}")
    return poses


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
    static_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(disparity) & (disparity > min_disparity)
    if static_mask is not None:
        if static_mask.shape != disparity.shape:
            raise ValueError("Dynamic mask must match the stereo disparity shape")
        valid &= static_mask
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
) -> Tuple[np.ndarray, np.ndarray]:
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
        mask_root = getattr(args, "dynamic_mask_root", None)
        training_cameras = getattr(args, "training_cameras", "left")
        image_name = f"image_00/{frame_id:010d}.png" if training_cameras == "stereo" else f"{frame_id:010d}.png"
        static_mask = read_static_mask(mask_root, image_name, left.shape[:2]) if mask_root else None
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
            static_mask=static_mask,
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
    if points.shape[0] > args.stereo_max_points:
        keep = rng.choice(points.shape[0], size=args.stereo_max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    return points, colors


def build_random_points(
    frames: Sequence[Tuple[int, Path, Path, FramePose]],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.random_seed)
    camera_centers = np.stack([frame_pose.c2w[:3, 3] for _, _, _, frame_pose in frames], axis=0)
    center_min = camera_centers.min(axis=0)
    center_max = camera_centers.max(axis=0)
    span = np.maximum(center_max - center_min, 1.0)
    num_points = min(args.stereo_max_points, max(5000, len(frames) * 256))
    points = rng.uniform(center_min - 0.5 * span, center_max + 0.5 * span, size=(num_points, 3))
    colors = rng.integers(0, 255, size=(num_points, 3), dtype=np.uint8)
    return points.astype(np.float32), colors


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
    frame_split_by_id: dict[int, str],
) -> tuple[list[ColmapImage], list[dict]]:
    colmap_images: list[ColmapImage] = []
    frame_records: list[dict] = []
    image_id = 1

    for frame_id, left_path, right_path, pose in frames:
        output_name = f"{frame_id:010d}.png"
        frame_images: list[dict] = []
        split_name = frame_split_by_id[int(frame_id)]

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
            if split_name == "train":
                colmap_images.append(
                    ColmapImage(
                        image_id=image_id,
                        camera_id=camera_id,
                        image_name=image_name,
                        c2w=c2w,
                    )
                )
                image_id += 1
            frame_images.append(
                {
                    "camera": camera_label,
                    "camera_id": camera_id,
                    "image_name": image_name,
                    "source_path": str(src_path),
                    "c2w": matrix_for_metadata(c2w),
                }
            )

        frame_records.append({"frame_id": frame_id, "split": split_name, "images": frame_images})

    return colmap_images, frame_records


def materialize_dynamic_masks(
    *,
    frames: Sequence[Tuple[int, Path, Path, FramePose]],
    training_cameras: str,
    mask_root: Path | None,
    masks_out: Path,
    copy_mode: str,
) -> None:
    if mask_root is None:
        return
    load_manifest(mask_root)
    for frame_id, left_path, right_path, _pose in frames:
        filename = f"{frame_id:010d}.png"
        names = [filename] if training_cameras == "left" else [f"image_00/{filename}", f"image_01/{filename}"]
        for name in names:
            source = mask_path(mask_root, name)
            if not source.is_file():
                raise FileNotFoundError(f"Dynamic mask missing for prepared image {name}: {source}")
            try:
                import cv2
                image_path = left_path if name == filename or name.startswith("image_00/") else right_path
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
                if image is not None and (mask is None or mask.shape != image.shape):
                    raise ValueError(f"Dynamic mask {source} must match image {image_path}")
            except ImportError:
                pass
            materialize_image(source, masks_out / name, copy_mode)


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
    selected_frame_ids = [frame_id for frame_id, _, _, _ in frames]
    frame_splits = split_frame_indices(selected_frame_ids)
    frame_split_by_id = split_lookup(frame_splits)
    train_frames = [
        frame
        for frame in frames
        if frame_split_by_id[int(frame[0])] == "train"
    ]

    dataset_dir = args.output_root / args.drive
    images_out = dataset_dir / "images"
    masks_out = dataset_dir / "masks"
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
        frame_split_by_id=frame_split_by_id,
    )
    mask_root = getattr(args, "dynamic_mask_root", None)
    materialize_dynamic_masks(
        frames=frames,
        training_cameras=training_cameras,
        mask_root=mask_root,
        masks_out=masks_out,
        copy_mode=args.copy_mode,
    )

    if args.seed_mode == "stereo":
        points_xyz, points_rgb = build_sparse_points_from_stereo(train_frames, calibration, args)
    else:
        points_xyz, points_rgb = build_random_points(train_frames, args)

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
        "frame_splits": frame_splits,
        "split_counts": {
            split_name: len(frame_ids)
            for split_name, frame_ids in frame_splits.items()
        },
        "training_cameras": training_cameras,
        "frame_step": args.frame_step,
        "max_frames": args.max_frames,
        "copy_mode": args.copy_mode,
        "seed_mode": args.seed_mode,
        "dynamic_mask_root": str(mask_root) if mask_root else None,
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
        "selected_frames": selected_frame_ids,
        "frame_records": frame_records,
    }
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
