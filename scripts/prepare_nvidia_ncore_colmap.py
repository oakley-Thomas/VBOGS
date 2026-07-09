#!/usr/bin/env python3

"""Prepare an NVIDIA PhysicalAI AV NCore clip for Octree-AnyGS.

The script decodes selected NCore camera frames into Octree-AnyGS's
COLMAP-style `images/` + `sparse/0/` layout and writes a sparse seed point
cloud from NCore LiDAR when available.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.data_layout import resolve_nvidia_ncore_root
from vbogs.ncore_adapter import (
    DEFAULT_CAMERA_DEPTH_PAIR,
    DEFAULT_CAMERA_IDS,
    DEFAULT_LIDAR_ID,
    ColmapImage,
    PinholeCamera,
    closest_frame_index,
    extract_pinhole_camera,
    get_frame_count,
    get_frame_timestamp_us,
    get_image_array,
    get_sensor_c2w,
    json_ready_matrix,
    load_ncore_sequence_loader,
    parse_id_list,
    rectify_camera_pair,
    save_metadata,
    selected_frame_indices,
    transform_points,
    unproject_rectified_to_world,
    write_cameras_txt,
    write_images_txt,
    write_image,
    write_ply,
)


@dataclass(frozen=True)
class PreparedDataset:
    dataset_dir: Path
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-id",
        required=True,
        help="NCore clip/sequence id. Converted outputs are written under this id.",
    )
    parser.add_argument(
        "--ncore-root",
        type=Path,
        default=None,
        help="Root containing converted NCore V4 clips. Defaults to repo/container data paths.",
    )
    parser.add_argument(
        "--camera-id",
        dest="camera_ids",
        action="append",
        default=None,
        help=(
            "Camera sensor id to include. Repeat or pass comma-separated ids. "
            "Defaults to camera_front_wide_120fov."
        ),
    )
    parser.add_argument(
        "--lidar-id",
        default=DEFAULT_LIDAR_ID,
        help="LiDAR sensor id used for the sparse seed point cloud.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/COLMAP"),
        help="Root directory for prepared Octree-AnyGS datasets.",
    )
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="copy",
        help=(
            "Accepted for pipeline compatibility. NCore frames are decoded, so "
            "both modes write image files into the prepared dataset."
        ),
    )
    parser.add_argument(
        "--seed-mode",
        choices=("lidar", "stereo", "random"),
        default="lidar",
        help="How to bootstrap sparse points for Octree-AnyGS ingest.",
    )
    parser.add_argument("--seed-max-points", type=int, default=60000)
    parser.add_argument("--max-points-per-lidar-frame", type=int, default=5000)
    parser.add_argument(
        "--seed-stereo-pair",
        default=",".join(DEFAULT_CAMERA_DEPTH_PAIR),
        help="Comma-separated left,right NCore camera ids for `--seed-mode stereo`.",
    )
    parser.add_argument("--max-points-per-stereo-frame", type=int, default=5000)
    parser.add_argument("--seed-pixel-step", type=int, default=8)
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--min-disparity", type=float, default=2.0)
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _point_cloud_xyz(point_cloud: Any) -> np.ndarray:
    for name in ("xyz_m_end", "xyz", "points", "positions"):
        if hasattr(point_cloud, name):
            xyz = np.asarray(getattr(point_cloud, name), dtype=np.float32)
            if xyz.ndim == 2 and xyz.shape[1] >= 3:
                return xyz[:, :3]
    raise ValueError("Could not find XYZ coordinates on NCore LiDAR point cloud")


def get_lidar_points_world(lidar_sensor: Any, frame_index: int) -> np.ndarray:
    if hasattr(lidar_sensor, "get_frame_point_cloud"):
        try:
            point_cloud = lidar_sensor.get_frame_point_cloud(
                int(frame_index),
                motion_compensation=True,
                with_start_points=False,
            )
        except TypeError:
            point_cloud = lidar_sensor.get_frame_point_cloud(int(frame_index))
    else:
        point_cloud = lidar_sensor.get_pc(int(frame_index))
    points_sensor = _point_cloud_xyz(point_cloud)
    return transform_points(points_sensor, get_sensor_c2w(lidar_sensor, frame_index))


def build_sparse_seed_from_lidar(
    *,
    loader: Any,
    lidar_id: str,
    frame_step: int,
    max_frames: int,
    seed_max_points: int,
    max_points_per_lidar_frame: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    lidar_sensor = loader.get_lidar_sensor(lidar_id)
    rng = np.random.default_rng(random_seed)
    frame_indices = selected_frame_indices(lidar_sensor, frame_step, max_frames)

    all_xyz: list[np.ndarray] = []
    for frame_index in frame_indices:
        xyz = get_lidar_points_world(lidar_sensor, frame_index)
        if max_points_per_lidar_frame > 0 and xyz.shape[0] > max_points_per_lidar_frame:
            keep = rng.choice(xyz.shape[0], size=max_points_per_lidar_frame, replace=False)
            xyz = xyz[keep]
        if xyz.size:
            all_xyz.append(xyz.astype(np.float32, copy=False))

    if not all_xyz:
        raise RuntimeError("NCore LiDAR seed export produced no points")

    xyz = np.concatenate(all_xyz, axis=0)
    if seed_max_points > 0 and xyz.shape[0] > seed_max_points:
        keep = rng.choice(xyz.shape[0], size=seed_max_points, replace=False)
        xyz = xyz[keep]
    rgb = np.full((xyz.shape[0], 3), 160, dtype=np.uint8)
    return xyz, rgb, {
        "seed_lidar_id": lidar_id,
        "seed_lidar_frames": frame_indices,
        "seed_point_count": int(xyz.shape[0]),
    }


def build_sparse_seed_from_stereo(
    *,
    loader: Any,
    pair: Sequence[str],
    frame_step: int,
    max_frames: int,
    seed_max_points: int,
    max_points_per_frame: int,
    pixel_step: int,
    num_disparities: int,
    block_size: int,
    min_disparity: float,
    max_depth_m: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import cv2

    if len(pair) != 2:
        raise ValueError("--seed-stereo-pair must resolve to exactly two camera ids")
    if num_disparities % 16 != 0:
        raise ValueError("--num-disparities must be divisible by 16")
    left_id, right_id = pair
    left_sensor = loader.get_camera_sensor(left_id)
    right_sensor = loader.get_camera_sensor(right_id)
    rng = np.random.default_rng(random_seed)
    frame_indices = selected_frame_indices(left_sensor, frame_step, max_frames)

    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    all_xyz: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    for left_index in frame_indices:
        timestamp_us = get_frame_timestamp_us(left_sensor, left_index)
        right_index = closest_frame_index(right_sensor, timestamp_us)
        left_rgb = get_image_array(left_sensor, left_index)
        right_rgb = get_image_array(right_sensor, right_index)
        left_camera = extract_pinhole_camera(
            camera_id=left_id,
            camera_model_id=1,
            model_parameters=left_sensor.model_parameters,
            image_shape=left_rgb.shape,
        )
        right_camera = extract_pinhole_camera(
            camera_id=right_id,
            camera_model_id=2,
            model_parameters=right_sensor.model_parameters,
            image_shape=right_rgb.shape,
        )
        left_c2w = get_sensor_c2w(left_sensor, left_index)
        right_c2w = get_sensor_c2w(right_sensor, right_index)
        left_rect, right_rect, p1, p2, r1 = rectify_camera_pair(
            left_rgb=left_rgb,
            right_rgb=right_rgb,
            left_camera=left_camera,
            right_camera=right_camera,
            left_c2w=left_c2w,
            right_c2w=right_c2w,
        )
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_RGB2GRAY)
        disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        xyz, rgb = unproject_rectified_to_world(
            disparity=disparity,
            left_rect_rgb=left_rect,
            p_left=p1,
            p_right=p2,
            r_rect_left=r1,
            left_c2w=left_c2w,
            min_disparity=min_disparity,
            max_depth_m=max_depth_m,
            pixel_step=pixel_step,
            max_points_per_frame=max_points_per_frame,
            rng=rng,
        )
        print(f"[stereo-seed] frame {left_index}: kept {xyz.shape[0]} seed points")
        if xyz.size:
            all_xyz.append(xyz)
            all_rgb.append(rgb)

    if not all_xyz:
        raise RuntimeError("NCore stereo seed export produced no points")

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    if seed_max_points > 0 and xyz.shape[0] > seed_max_points:
        keep = rng.choice(xyz.shape[0], size=seed_max_points, replace=False)
        xyz = xyz[keep]
        rgb = rgb[keep]
    return xyz, rgb, {
        "seed_stereo_pair": [left_id, right_id],
        "seed_frames": frame_indices,
        "seed_point_count": int(xyz.shape[0]),
    }


def build_random_seed(
    *,
    frame_records: Sequence[dict[str, Any]],
    seed_max_points: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    centers = []
    for record in frame_records:
        primary = record["cameras"][record["primary_camera_id"]]
        centers.append(np.asarray(primary["c2w"], dtype=np.float32)[:3, 3])
    centers_arr = np.stack(centers, axis=0)
    center_min = centers_arr.min(axis=0)
    center_max = centers_arr.max(axis=0)
    span = np.maximum(center_max - center_min, 1.0)
    count = min(seed_max_points, max(5000, len(frame_records) * 512))
    xyz = rng.uniform(center_min - 0.5 * span, center_max + 0.5 * span, size=(count, 3))
    rgb = rng.integers(0, 255, size=(count, 3), dtype=np.uint8)
    return xyz.astype(np.float32), rgb, {"seed_point_count": int(count)}


def prepare_dataset_from_loader(args: argparse.Namespace, loader: Any) -> PreparedDataset:
    camera_ids = parse_id_list(args.camera_ids, DEFAULT_CAMERA_IDS)
    missing = [camera_id for camera_id in camera_ids if camera_id not in loader.camera_ids]
    if missing:
        available = ", ".join(str(camera_id) for camera_id in loader.camera_ids)
        raise KeyError(
            "NCore sequence is missing requested cameras: "
            f"{', '.join(missing)}. Available cameras: {available}"
        )

    cameras_by_id = {camera_id: loader.get_camera_sensor(camera_id) for camera_id in camera_ids}
    primary_camera_id = camera_ids[0]
    primary_sensor = cameras_by_id[primary_camera_id]
    primary_indices = selected_frame_indices(primary_sensor, args.frame_step, args.max_frames)

    dataset_dir = Path(args.output_root) / args.scene_id
    images_out = dataset_dir / "images"
    sparse_out = dataset_dir / "sparse" / "0"
    ensure_empty_dir(dataset_dir)
    images_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    colmap_cameras: list[PinholeCamera] = []
    camera_model_ids = {camera_id: idx for idx, camera_id in enumerate(camera_ids, start=1)}
    colmap_images: list[ColmapImage] = []
    frame_records: list[dict[str, Any]] = []
    image_id = 1

    for primary_index in primary_indices:
        timestamp_us = get_frame_timestamp_us(primary_sensor, primary_index)
        record: dict[str, Any] = {
            "frame_id": int(primary_index),
            "timestamp_us": int(timestamp_us),
            "primary_camera_id": primary_camera_id,
            "cameras": {},
        }
        for camera_id, sensor in cameras_by_id.items():
            frame_index = primary_index if camera_id == primary_camera_id else closest_frame_index(sensor, timestamp_us)
            image_rgb = get_image_array(sensor, frame_index)
            camera_model_id = camera_model_ids[camera_id]
            if not any(camera.camera_id == camera_id for camera in colmap_cameras):
                colmap_cameras.append(
                    extract_pinhole_camera(
                        camera_id=camera_id,
                        camera_model_id=camera_model_id,
                        model_parameters=sensor.model_parameters,
                        image_shape=image_rgb.shape,
                    )
                )

            image_name = (
                f"{camera_id}/"
                f"{camera_id}_{int(primary_index):010d}_{int(frame_index):010d}.png"
            )
            write_image(images_out / image_name, image_rgb)
            c2w = get_sensor_c2w(sensor, frame_index)
            colmap_images.append(
                ColmapImage(
                    image_id=image_id,
                    camera_model_id=camera_model_id,
                    image_name=image_name,
                    c2w=c2w,
                )
            )
            record["cameras"][camera_id] = {
                "frame_index": int(frame_index),
                "timestamp_us": get_frame_timestamp_us(sensor, frame_index),
                "image_name": image_name,
                "c2w": json_ready_matrix(c2w),
            }
            image_id += 1
        frame_records.append(record)

    if args.seed_mode == "lidar":
        seed_xyz, seed_rgb, seed_metadata = build_sparse_seed_from_lidar(
            loader=loader,
            lidar_id=args.lidar_id,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            seed_max_points=args.seed_max_points,
            max_points_per_lidar_frame=args.max_points_per_lidar_frame,
            random_seed=args.random_seed,
        )
    elif args.seed_mode == "stereo":
        seed_xyz, seed_rgb, seed_metadata = build_sparse_seed_from_stereo(
            loader=loader,
            pair=parse_id_list(args.seed_stereo_pair, DEFAULT_CAMERA_DEPTH_PAIR),
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            seed_max_points=args.seed_max_points,
            max_points_per_frame=args.max_points_per_stereo_frame,
            pixel_step=args.seed_pixel_step,
            num_disparities=args.num_disparities,
            block_size=args.block_size,
            min_disparity=args.min_disparity,
            max_depth_m=args.max_depth_m,
            random_seed=args.random_seed,
        )
    else:
        seed_xyz, seed_rgb, seed_metadata = build_random_seed(
            frame_records=frame_records,
            seed_max_points=args.seed_max_points,
            random_seed=args.random_seed,
        )

    write_cameras_txt(sparse_out / "cameras.txt", colmap_cameras)
    write_images_txt(sparse_out / "images.txt", colmap_images)
    write_ply(sparse_out / "points3D.ply", seed_xyz, seed_rgb)

    metadata = {
        "dataset": "nvidia_ncore",
        "scene_id": args.scene_id,
        "ncore_root": str(args.ncore_root),
        "num_frames": len(frame_records),
        "num_images": len(colmap_images),
        "selected_frames": [record["frame_id"] for record in frame_records],
        "camera_ids": camera_ids,
        "primary_camera_id": primary_camera_id,
        "lidar_id": args.lidar_id,
        "frame_step": args.frame_step,
        "max_frames": args.max_frames,
        "copy_mode": args.copy_mode,
        "seed_mode": args.seed_mode,
        "seed_metadata": seed_metadata,
        "frame_records": frame_records,
        "intrinsics": {
            camera.camera_id: {
                "camera_model_id": camera.camera_model_id,
                "width": camera.width,
                "height": camera.height,
                "fx": camera.fx,
                "fy": camera.fy,
                "cx": camera.cx,
                "cy": camera.cy,
            }
            for camera in colmap_cameras
        },
    }
    save_metadata(dataset_dir / "metadata.json", metadata)
    return PreparedDataset(dataset_dir=dataset_dir, metadata=metadata)


def main() -> None:
    args = parse_args()
    args.ncore_root = resolve_nvidia_ncore_root(args.ncore_root)
    loader = load_ncore_sequence_loader(args.ncore_root, args.scene_id)
    prepared = prepare_dataset_from_loader(args, loader)
    print(f"Prepared NVIDIA NCore Octree-AnyGS dataset at: {prepared.dataset_dir}")


if __name__ == "__main__":
    main()
