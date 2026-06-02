#!/usr/bin/env python3

"""Export VBOGS world-frame point observations from supported datasets.

This is the generic Stage 2 entry point. KITTI-360 keeps the existing stereo
implementation, while NVIDIA NCore clips can export LiDAR-colored points or a
camera-depth stereo-style point cloud. The output contract remains:

- `xyz`: `(N, 3)` float32 world-frame points
- `rgb`: `(N, 3)` uint8 colors in RGB order
- `frame_id`: `(N,)` int32 source frame ids
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stereo_to_pointcloud
from scripts.prepare_nvidia_ncore_colmap import get_lidar_points_world
from vbogs.data_layout import resolve_nvidia_ncore_root
from vbogs.ncore_adapter import (
    DEFAULT_CAMERA_DEPTH_PAIR,
    DEFAULT_CAMERA_IDS,
    DEFAULT_LIDAR_ID,
    PinholeCamera,
    closest_frame_index,
    extract_pinhole_camera,
    get_frame_timestamp_us,
    get_image_array,
    get_sensor_c2w,
    load_ncore_sequence_loader,
    parse_id_list,
    selected_frame_indices,
    write_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        choices=("kitti360", "nvidia_ncore"),
        default="kitti360",
    )
    parser.add_argument("--scene-id", default=None, help="Dataset scene/clip id.")
    parser.add_argument(
        "--drive",
        default=None,
        help="Backward-compatible KITTI-360 drive id alias for --scene-id.",
    )
    parser.add_argument(
        "--point-source",
        choices=("stereo", "lidar", "camera_depth"),
        default=None,
        help="Point source. Defaults to stereo for KITTI-360 and lidar for NVIDIA NCore.",
    )

    kitti_group = parser.add_argument_group("KITTI-360 stereo inputs")
    kitti_group.add_argument("--raw-root", type=Path, default=None)
    kitti_group.add_argument("--poses-root", type=Path, default=None)
    kitti_group.add_argument("--calibration-dir", type=Path, default=None)

    ncore_group = parser.add_argument_group("NVIDIA NCore inputs")
    ncore_group.add_argument("--ncore-root", type=Path, default=None)
    ncore_group.add_argument(
        "--camera-id",
        dest="camera_ids",
        action="append",
        default=None,
        help="Camera id used for coloring/training. Repeat or pass comma-separated ids.",
    )
    ncore_group.add_argument(
        "--camera-depth-pair",
        default=",".join(DEFAULT_CAMERA_DEPTH_PAIR),
        help="Comma-separated left,right NCore camera ids for camera_depth export.",
    )
    ncore_group.add_argument("--lidar-id", default=DEFAULT_LIDAR_ID)

    parser.add_argument(
        "--selection-metadata",
        type=Path,
        default=None,
        help="Prepared COLMAP metadata.json used to align exported point frames.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/points_world"))
    parser.add_argument("--output-name", default="points_world.npz")
    parser.add_argument("--write-ply", action="store_true")
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--pixel-step", type=int, default=1)
    parser.add_argument("--max-points-per-frame", type=int, default=250000)
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--min-disparity", type=float, default=2.0)
    parser.add_argument("--matcher", choices=("sgbm", "raft"), default="sgbm")
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--matcher-min-disparity", type=int, default=0)
    parser.add_argument("--lr-consistency-threshold", type=float, default=1.0)
    parser.add_argument("--texture-window-size", type=int, default=9)
    parser.add_argument("--texture-threshold", type=float, default=5.0)
    parser.add_argument("--uniqueness-ratio", type=int, default=10)
    parser.add_argument("--speckle-window-size", type=int, default=50)
    parser.add_argument("--speckle-range", type=int, default=2)
    parser.add_argument("--disp12-max-diff", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def scene_id(args: argparse.Namespace) -> str:
    value = args.scene_id or args.drive
    if not value:
        raise ValueError("--scene-id is required unless --drive is provided")
    return value


def point_source(args: argparse.Namespace) -> str:
    if args.point_source:
        return args.point_source
    return "stereo" if args.dataset_name == "kitti360" else "lidar"


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def point_output_paths(args: argparse.Namespace, scene: str) -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_root) / scene
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    metadata_path = output_dir / f"{output_path.stem}_metadata.json"
    ply_path = output_dir / f"{output_path.stem}.ply"
    return output_path, metadata_path, ply_path


def write_points_artifact(
    *,
    args: argparse.Namespace,
    scene: str,
    xyz: np.ndarray,
    rgb: np.ndarray,
    frame_id: np.ndarray,
    metadata: dict[str, Any],
) -> Path:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    frame_id = np.asarray(frame_id, dtype=np.int32).reshape(-1)
    if xyz.shape[0] != rgb.shape[0] or xyz.shape[0] != frame_id.shape[0]:
        raise ValueError("xyz, rgb, and frame_id must have matching first dimensions")
    if xyz.shape[0] == 0:
        raise RuntimeError("Point export produced no world points")

    output_path, metadata_path, ply_path = point_output_paths(args, scene)
    np.savez_compressed(output_path, xyz=xyz, rgb=rgb, frame_id=frame_id)
    if args.write_ply:
        write_ply(ply_path, xyz, rgb)
    metadata = {
        **metadata,
        "dataset": args.dataset_name,
        "scene_id": scene,
        "output_npz": str(output_path),
        "output_ply": str(ply_path) if args.write_ply else None,
        "num_points": int(xyz.shape[0]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def camera_spec(sensor: Any, camera_id: str, image_rgb: np.ndarray, camera_model_id: int = 1) -> PinholeCamera:
    return extract_pinhole_camera(
        camera_id=camera_id,
        camera_model_id=camera_model_id,
        model_parameters=sensor.model_parameters,
        image_shape=image_rgb.shape,
    )


def sample_camera_colors_pinhole(
    *,
    points_world: np.ndarray,
    image_rgb: np.ndarray,
    camera: PinholeCamera,
    c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64).reshape(4, 4))
    hom = np.concatenate([points_world, np.ones((points_world.shape[0], 1))], axis=1)
    points_cam = (w2c @ hom.T).T[:, :3]
    z = points_cam[:, 2]
    valid = z > 1e-6
    u = camera.fx * (points_cam[:, 0] / np.maximum(z, 1e-6)) + camera.cx
    v = camera.fy * (points_cam[:, 1] / np.maximum(z, 1e-6)) + camera.cy
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    valid &= (ui >= 0) & (ui < image_rgb.shape[1]) & (vi >= 0) & (vi < image_rgb.shape[0])

    colors = np.zeros((points_world.shape[0], 3), dtype=np.uint8)
    if np.any(valid):
        colors[valid] = image_rgb[vi[valid], ui[valid], :3]
    return colors, valid


def color_points_from_cameras(
    *,
    loader: Any,
    points_world: np.ndarray,
    timestamp_us: int,
    camera_ids: Sequence[str],
    default_color: int = 160,
) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
    colors = np.full((points_world.shape[0], 3), default_color, dtype=np.uint8)
    missing = np.ones(points_world.shape[0], dtype=bool)
    stats: list[dict[str, Any]] = []
    for camera_idx, camera_id in enumerate(camera_ids, start=1):
        sensor = loader.get_camera_sensor(camera_id)
        frame_index = closest_frame_index(sensor, timestamp_us)
        image_rgb = get_image_array(sensor, frame_index)
        spec = camera_spec(sensor, camera_id, image_rgb, camera_model_id=camera_idx)
        sampled, valid = sample_camera_colors_pinhole(
            points_world=points_world,
            image_rgb=image_rgb,
            camera=spec,
            c2w=get_sensor_c2w(sensor, frame_index),
        )
        assign = missing & valid
        colors[assign] = sampled[assign]
        missing[assign] = False
        stats.append(
            {
                "camera_id": camera_id,
                "frame_index": int(frame_index),
                "timestamp_us": get_frame_timestamp_us(sensor, frame_index),
                "colored_points": int(np.count_nonzero(assign)),
                "valid_projected_points": int(np.count_nonzero(valid)),
            }
        )
        if not np.any(missing):
            break
    return colors, int(colors.shape[0] - np.count_nonzero(missing)), stats


def export_lidar_points_from_loader(args: argparse.Namespace, loader: Any) -> Path:
    scene = scene_id(args)
    camera_ids = parse_id_list(args.camera_ids, DEFAULT_CAMERA_IDS)
    lidar_sensor = loader.get_lidar_sensor(args.lidar_id)
    frame_indices = selected_frame_indices(lidar_sensor, args.frame_step, args.max_frames)
    rng = np.random.default_rng(args.random_seed)

    all_xyz: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    all_frame_ids: list[np.ndarray] = []
    frame_stats: list[dict[str, Any]] = []

    for frame_index in frame_indices:
        timestamp_us = get_frame_timestamp_us(lidar_sensor, frame_index)
        xyz = get_lidar_points_world(lidar_sensor, frame_index)
        source_count = int(xyz.shape[0])
        if args.max_points_per_frame > 0 and xyz.shape[0] > args.max_points_per_frame:
            keep = rng.choice(xyz.shape[0], size=args.max_points_per_frame, replace=False)
            xyz = xyz[keep]

        colors, colored_count, color_stats = color_points_from_cameras(
            loader=loader,
            points_world=xyz,
            timestamp_us=timestamp_us,
            camera_ids=camera_ids,
        )
        all_xyz.append(xyz.astype(np.float32, copy=False))
        all_rgb.append(colors)
        all_frame_ids.append(np.full(xyz.shape[0], frame_index, dtype=np.int32))
        frame_stats.append(
            {
                "frame_id": int(frame_index),
                "timestamp_us": int(timestamp_us),
                "source_points": source_count,
                "points_kept": int(xyz.shape[0]),
                "colored_points": colored_count,
                "camera_color_stats": color_stats,
            }
        )
        print(
            f"[ncore-lidar] frame {frame_index}: "
            f"kept={xyz.shape[0]} colored={colored_count}"
        )

    metadata = {
        "point_source": "lidar",
        "lidar_id": args.lidar_id,
        "camera_ids": camera_ids,
        "num_frames": len(frame_indices),
        "frame_step": args.frame_step,
        "max_frames": args.max_frames,
        "max_points_per_frame": args.max_points_per_frame,
        "frame_stats": frame_stats,
    }
    return write_points_artifact(
        args=args,
        scene=scene,
        xyz=np.concatenate(all_xyz, axis=0),
        rgb=np.concatenate(all_rgb, axis=0),
        frame_id=np.concatenate(all_frame_ids, axis=0),
        metadata=metadata,
    )


def rectify_camera_pair(
    *,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_camera: PinholeCamera,
    right_camera: PinholeCamera,
    left_c2w: np.ndarray,
    right_c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    image_size = (left_camera.width, left_camera.height)
    k_left = np.array(
        [[left_camera.fx, 0.0, left_camera.cx], [0.0, left_camera.fy, left_camera.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    k_right = np.array(
        [[right_camera.fx, 0.0, right_camera.cx], [0.0, right_camera.fy, right_camera.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    t_left_right = np.linalg.inv(right_c2w) @ left_c2w
    r_lr = t_left_right[:3, :3]
    t_lr = t_left_right[:3, 3]
    zeros = np.zeros((5, 1), dtype=np.float64)
    r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        k_left,
        zeros,
        k_right,
        zeros,
        image_size,
        r_lr,
        t_lr,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(k_left, zeros, r1, p1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(k_right, zeros, r2, p2, image_size, cv2.CV_32FC1)
    left_rect = cv2.remap(left_rgb, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_rgb, map2x, map2y, cv2.INTER_LINEAR)
    return left_rect, right_rect, p1, p2, r1


def unproject_rectified_to_world(
    *,
    disparity: np.ndarray,
    left_rect_rgb: np.ndarray,
    p_left: np.ndarray,
    p_right: np.ndarray,
    r_rect_left: np.ndarray,
    left_c2w: np.ndarray,
    min_disparity: float,
    max_depth_m: float,
    pixel_step: int,
    max_points_per_frame: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(disparity) & (disparity > min_disparity)
    if pixel_step > 1:
        yy = np.arange(disparity.shape[0])[:, None]
        xx = np.arange(disparity.shape[1])[None, :]
        valid &= (yy % pixel_step == 0) & (xx % pixel_step == 0)
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    fx = float(p_left[0, 0])
    fy = float(p_left[1, 1])
    cx = float(p_left[0, 2])
    cy = float(p_left[1, 2])
    baseline = abs(float(p_right[0, 3]) / max(float(p_right[0, 0]), 1e-6))
    depth = (fx * baseline) / disparity[ys, xs]
    depth_valid = np.isfinite(depth) & (depth > 0.0)
    if max_depth_m > 0:
        depth_valid &= depth <= max_depth_m
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

    x_rect = (xs.astype(np.float64) - cx) * depth / fx
    y_rect = (ys.astype(np.float64) - cy) * depth / fy
    z_rect = depth.astype(np.float64)
    points_rect = np.stack([x_rect, y_rect, z_rect], axis=1)
    points_left = (np.asarray(r_rect_left, dtype=np.float64).T @ points_rect.T).T
    hom_left = np.concatenate([points_left, np.ones((points_left.shape[0], 1))], axis=1)
    points_world = (np.asarray(left_c2w, dtype=np.float64) @ hom_left.T).T[:, :3].astype(np.float32)
    colors = left_rect_rgb[ys, xs, :3].astype(np.uint8)
    return points_world, colors


def export_camera_depth_from_loader(args: argparse.Namespace, loader: Any) -> Path:
    import cv2

    scene = scene_id(args)
    pair = parse_id_list(args.camera_depth_pair, DEFAULT_CAMERA_DEPTH_PAIR)
    if len(pair) != 2:
        raise ValueError("--camera-depth-pair must resolve to exactly two camera ids")
    left_id, right_id = pair
    left_sensor = loader.get_camera_sensor(left_id)
    right_sensor = loader.get_camera_sensor(right_id)
    selection_metadata = load_json(args.selection_metadata)
    selected_frames = selection_metadata.get("selected_frames")
    if isinstance(selected_frames, list) and selected_frames:
        frame_indices = [int(frame_id) for frame_id in selected_frames]
        if args.max_frames:
            frame_indices = frame_indices[: args.max_frames]
    else:
        frame_indices = selected_frame_indices(left_sensor, args.frame_step, args.max_frames)
    matcher = stereo_to_pointcloud.build_matcher(args)
    rng = np.random.default_rng(args.random_seed)

    all_xyz: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    all_frame_ids: list[np.ndarray] = []
    frame_stats: list[dict[str, Any]] = []

    for left_index in frame_indices:
        timestamp_us = get_frame_timestamp_us(left_sensor, left_index)
        right_index = closest_frame_index(right_sensor, timestamp_us)
        left_rgb = get_image_array(left_sensor, left_index)
        right_rgb = get_image_array(right_sensor, right_index)
        left_cam = camera_spec(left_sensor, left_id, left_rgb, camera_model_id=1)
        right_cam = camera_spec(right_sensor, right_id, right_rgb, camera_model_id=2)
        left_c2w = get_sensor_c2w(left_sensor, left_index)
        right_c2w = get_sensor_c2w(right_sensor, right_index)
        left_rect, right_rect, p1, p2, r1 = rectify_camera_pair(
            left_rgb=left_rgb,
            right_rgb=right_rgb,
            left_camera=left_cam,
            right_camera=right_cam,
            left_c2w=left_c2w,
            right_c2w=right_c2w,
        )
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_RGB2GRAY)
        stereo = matcher.compute(left_gray, right_gray)
        points_world, colors = unproject_rectified_to_world(
            disparity=stereo.disparity_left,
            left_rect_rgb=left_rect,
            p_left=p1,
            p_right=p2,
            r_rect_left=r1,
            left_c2w=left_c2w,
            min_disparity=args.min_disparity,
            max_depth_m=args.max_depth_m,
            pixel_step=args.pixel_step,
            max_points_per_frame=args.max_points_per_frame,
            rng=rng,
        )
        frame_stats.append(
            {
                "frame_id": int(left_index),
                "left_camera_id": left_id,
                "right_camera_id": right_id,
                "right_frame_index": int(right_index),
                "timestamp_us": int(timestamp_us),
                "points_kept": int(points_world.shape[0]),
            }
        )
        print(f"[ncore-camera-depth] frame {left_index}: kept={points_world.shape[0]}")
        if points_world.size:
            all_xyz.append(points_world)
            all_rgb.append(colors)
            all_frame_ids.append(np.full(points_world.shape[0], left_index, dtype=np.int32))

    metadata = {
        "point_source": "camera_depth",
        "camera_depth_pair": [left_id, right_id],
        "matcher": matcher.name,
        "num_frames": len(frame_indices),
        "frame_step": args.frame_step,
        "max_frames": args.max_frames,
        "filters": {
            "min_disparity": args.min_disparity,
            "max_depth_m": args.max_depth_m,
            "pixel_step": args.pixel_step,
            "max_points_per_frame": args.max_points_per_frame,
        },
        "frame_stats": frame_stats,
    }
    if not all_xyz:
        raise RuntimeError("NVIDIA NCore camera-depth export produced no valid world points")
    return write_points_artifact(
        args=args,
        scene=scene,
        xyz=np.concatenate(all_xyz, axis=0),
        rgb=np.concatenate(all_rgb, axis=0),
        frame_id=np.concatenate(all_frame_ids, axis=0),
        metadata=metadata,
    )


def export_kitti_stereo(args: argparse.Namespace) -> Path:
    args.drive = scene_id(args)
    stereo_to_pointcloud.resolve_input_layout(args)
    return stereo_to_pointcloud.export_points(args)


def main() -> None:
    args = parse_args()
    source = point_source(args)
    if args.dataset_name == "kitti360":
        if source != "stereo":
            raise ValueError("KITTI-360 currently supports only --point-source stereo")
        output_path = export_kitti_stereo(args)
    elif args.dataset_name == "nvidia_ncore":
        if source == "stereo":
            raise ValueError("NVIDIA NCore point sources are lidar or camera_depth")
        args.ncore_root = resolve_nvidia_ncore_root(args.ncore_root)
        loader = load_ncore_sequence_loader(args.ncore_root, scene_id(args))
        if source == "lidar":
            output_path = export_lidar_points_from_loader(args, loader)
        else:
            output_path = export_camera_depth_from_loader(args, loader)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    print(f"Wrote world point cloud to: {output_path}")


if __name__ == "__main__":
    main()
