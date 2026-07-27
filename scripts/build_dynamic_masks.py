#!/usr/bin/env python3
"""Build static-region alpha masks for KITTI-360 and NVIDIA NCore scenes.

This is deliberately an offline preprocessing step.  It never edits source
images and writes an inspectable mask artifact that both preparation adapters
consume later.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.dynamic_masking import (
    COCO_NAMES,
    InstanceObservation,
    TorchvisionMaskRCNNSegmenter,
    associate_world_tracks,
    dilate_dynamic_mask,
    is_confirmed_moving,
    track_motion_metrics,
    write_overlay,
    write_manifest,
    write_static_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", choices=("kitti360", "nvidia_ncore"), required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--dilation-pixels", type=int, default=5)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--min-span-s", type=float, default=0.5)
    parser.add_argument("--min-displacement-m", type=float, default=1.0)
    parser.add_argument("--min-speed-mps", type=float, default=0.5)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--poses-root", type=Path, default=None)
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--training-cameras", choices=("left", "stereo"), default="left")
    parser.add_argument("--ncore-root", type=Path, default=None)
    parser.add_argument("--camera-id", dest="camera_ids", action="append", default=None)
    return parser.parse_args()


def kitti_centroid(mask: np.ndarray, disparity: np.ndarray, calibration: Any, c2w: np.ndarray) -> np.ndarray | None:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(disparity) & (disparity > 2.0)
    ys, xs = np.nonzero(valid)
    if ys.size < 50:
        return None
    depth = (float(calibration.fx) * float(calibration.baseline_m)) / disparity[ys, xs]
    valid_depth = np.isfinite(depth) & (depth > 0.0) & (depth <= 80.0)
    if int(valid_depth.sum()) < 50:
        return None
    ys, xs, depth = ys[valid_depth], xs[valid_depth], depth[valid_depth]
    xyz_camera = np.stack(
        [
            (xs - calibration.cx) * depth / calibration.fx,
            (ys - calibration.cy) * depth / calibration.fy,
            depth,
            np.ones_like(depth),
        ],
        axis=1,
    )
    xyz_world = (np.asarray(c2w) @ xyz_camera.T).T[:, :3]
    return np.median(xyz_world, axis=0)


def _kitti_mask_name(training_cameras: str, camera: str, frame_id: int) -> str:
    name = f"{frame_id:010d}.png"
    return f"{camera}/{name}" if training_cameras == "stereo" else name


def build_kitti(args: argparse.Namespace, segmenter: TorchvisionMaskRCNNSegmenter, mask_root: Path) -> dict[str, Any]:
    import cv2
    from scripts import prepare_kitti360_colmap as prepare
    from scripts import stereo_to_pointcloud as stereo
    from vbogs.data_layout import resolve_kitti360_path

    raw_root = resolve_kitti360_path(args.raw_root, kind="raw")
    poses_root = resolve_kitti360_path(args.poses_root, kind="poses")
    calibration_dir = resolve_kitti360_path(args.calibration_dir, kind="calibration")
    calibration = prepare.parse_perspective_file(calibration_dir / "perspective.txt")
    poses = prepare.parse_cam0_to_world(poses_root / args.scene_id / "cam0_to_world.txt")
    drive_root = raw_root / args.scene_id
    frames = prepare.sample_drive_frames(
        drive_root / "image_00" / "data_rect", drive_root / "image_01" / "data_rect",
        poses, args.frame_step, args.max_frames,
    )
    matcher_args = argparse.Namespace(
        matcher_min_disparity=0, num_disparities=128, block_size=5, uniqueness_ratio=10,
        speckle_window_size=50, speckle_range=2, disp12_max_diff=1,
    )
    matcher = stereo.SGBMStereoMatcher(
        min_disparity=0, num_disparities=128, block_size=5, uniqueness_ratio=10,
        speckle_window_size=50, speckle_range=2, disp12_max_diff=1,
    )
    observations: list[InstanceObservation] = []
    per_frame: dict[int, list[tuple[int, float, np.ndarray]]] = {}
    disparities: dict[int, np.ndarray] = {}
    for frame_id, left_path, right_path, pose in frames:
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise RuntimeError(f"Could not read KITTI stereo pair {frame_id}")
        left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        per_frame[frame_id] = segmenter.detect(left_rgb)
        disparity = matcher.compute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)).disparity_left
        disparities[frame_id] = disparity
        for class_id, score, instance_mask in per_frame[frame_id]:
            observations.append(InstanceObservation(
                frame_key=str(frame_id), timestamp_s=float(frame_id) / 10.0, class_id=class_id,
                mask=instance_mask, centroid_world=kitti_centroid(instance_mask, disparity, calibration, pose.c2w), score=score,
            ))
    tracks = associate_world_tracks(observations)
    moving = [track for track in tracks if is_confirmed_moving(
        track, min_observations=args.min_observations, min_span_s=args.min_span_s,
        min_displacement_m=args.min_displacement_m, min_speed_mps=args.min_speed_mps,
    )]
    moving_track_ids = {track.track_id for track in moving}
    moving_observations = {id(item) for track in moving for item in track.observations}
    dynamic_by_frame: dict[int, np.ndarray] = {}
    frame_stats: list[dict[str, Any]] = []
    for frame_id, _, _, _ in frames:
        dynamic = np.zeros(disparities[frame_id].shape, dtype=bool)
        for observation in observations:
            if observation.frame_key == str(frame_id) and id(observation) in moving_observations:
                dynamic |= observation.mask
        dynamic = dilate_dynamic_mask(dynamic, args.dilation_pixels)
        dynamic_by_frame[frame_id] = dynamic
        frame_stats.append({"frame_id": int(frame_id), "dynamic_pixels": int(dynamic.sum()), "static_pixels": int((~dynamic).sum())})
        write_static_mask(mask_root, _kitti_mask_name(args.training_cameras, "image_00", frame_id), ~dynamic)
        left_path = next(frame[1] for frame in frames if frame[0] == frame_id)
        left_bgr = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        if left_bgr is not None:
            write_overlay(mask_root, _kitti_mask_name(args.training_cameras, "image_00", frame_id), cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB), dynamic)
        if args.training_cameras == "stereo":
            # Warp accepted left masks using the rectified disparity.  This preserves
            # the confirmed-mover policy for the optional right training camera.
            warped = np.zeros_like(dynamic)
            ys, xs = np.nonzero(dynamic)
            xr = np.rint(xs.astype(np.float32) - disparities[frame_id][ys, xs]).astype(int)
            valid = (xr >= 0) & (xr < dynamic.shape[1])
            warped[ys[valid], xr[valid]] = True
            write_static_mask(mask_root, _kitti_mask_name(args.training_cameras, "image_01", frame_id), ~dilate_dynamic_mask(warped, args.dilation_pixels))
    return {
        "dataset": "kitti360", "scene_id": args.scene_id, "motion_source": "stereo_world_tracks",
        "frame_count": len(frames),
        "static_pixel_fraction": float(np.mean([(~item).mean() for item in dynamic_by_frame.values()])),
        "frame_stats": frame_stats,
        "tracks": [
            {"track_id": track.track_id, "class": COCO_NAMES.get(track.class_id, str(track.class_id)),
             "confirmed_moving": track.track_id in moving_track_ids, **track_motion_metrics(track)} for track in tracks
        ],
    }


def _ncore_bbox_center_world(loader: Any, observation: Any) -> np.ndarray:
    timestamp_us = int(observation.timestamp_us)
    rig_to_world = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", np.asarray([timestamp_us], dtype=np.uint64)))[0]
    center = np.asarray(observation.bbox3.centroid, dtype=np.float64).reshape(3)
    return rig_to_world[:3, :3] @ center + rig_to_world[:3, 3]


def _rotation_from_euler_xyz(angles: np.ndarray) -> np.ndarray:
    """NCore ``BBox3.rot`` is a three-angle local rotation in radians."""
    x, y, z = np.asarray(angles, dtype=float).reshape(3)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def build_ncore(args: argparse.Namespace, segmenter: TorchvisionMaskRCNNSegmenter, mask_root: Path) -> dict[str, Any]:
    """Use NCore track motion as authority; mask detections overlapping projected track centers.

    NCore releases have varied bbox3 APIs.  The artifact records the fallback when
    extents/orientation are unavailable instead of making a false dynamic claim.
    """
    from vbogs.data_layout import resolve_nvidia_ncore_root
    from vbogs.ncore_adapter import (
        DEFAULT_CAMERA_IDS, closest_frame_index, extract_pinhole_camera,
        get_frame_timestamp_us, get_image_array, get_sensor_c2w,
        load_ncore_sequence_loader, parse_id_list, selected_frame_indices,
    )

    loader = load_ncore_sequence_loader(resolve_nvidia_ncore_root(args.ncore_root), args.scene_id)
    camera_ids = parse_id_list(args.camera_ids, DEFAULT_CAMERA_IDS)
    tracks: dict[str, list[Any]] = defaultdict(list)
    for observation in loader.get_cuboid_track_observations():
        tracks[str(observation.track_id)].append(observation)
    moving_ids: set[str] = set()
    moving_cuboids: list[dict[str, Any]] = []
    track_rows = []
    for track_id, rows in tracks.items():
        rows.sort(key=lambda row: int(row.timestamp_us))
        positions = np.asarray([_ncore_bbox_center_world(loader, row) for row in rows])
        times = np.asarray([int(row.timestamp_us) * 1e-6 for row in rows])
        temporary = [InstanceObservation(track_id, float(time), -1, np.zeros((1, 1), bool), position, 1.0) for time, position in zip(times, positions)]
        from vbogs.dynamic_masking import MotionTrack
        track = MotionTrack(len(track_rows), -1, temporary)
        confirmed = is_confirmed_moving(track, min_observations=args.min_observations, min_span_s=args.min_span_s, min_displacement_m=args.min_displacement_m, min_speed_mps=args.min_speed_mps)
        track_rows.append({"track_id": track_id, "confirmed_moving": confirmed, **track_motion_metrics(track)})
        if not confirmed:
            continue
        moving_ids.add(track_id)
        for row, center in zip(rows, positions):
            bbox = row.bbox3
            size = getattr(bbox, "size", getattr(bbox, "dimensions", getattr(bbox, "dim", getattr(bbox, "extent", None))))
            if size is None:
                continue
            size_arr = np.asarray(size, dtype=float).reshape(-1)[:3]
            if size_arr.size != 3:
                continue
            local_rotation = getattr(bbox, "rotation", getattr(bbox, "orientation", getattr(bbox, "rot", np.eye(3))))
            local_rotation = np.asarray(local_rotation, dtype=float)
            if local_rotation.size == 9:
                local_rotation = local_rotation.reshape(3, 3)
            elif local_rotation.size == 3:
                local_rotation = _rotation_from_euler_xyz(local_rotation)
            else:
                local_rotation = np.eye(3)
            rig_to_world = np.asarray(loader.pose_graph.evaluate_poses(
                "rig", "world", np.asarray([int(row.timestamp_us)], dtype=np.uint64)
            ))[0]
            moving_cuboids.append({
                "track_id": track_id, "timestamp_us": int(row.timestamp_us),
                "center_world": center.tolist(), "size_m": size_arr.tolist(),
                "rotation_world": (rig_to_world[:3, :3] @ local_rotation).tolist(),
            })
    # The centroid projection is portable across NCore releases. A detected mask
    # is accepted only if it contains the projected confirmed-track centroid.
    primary_sensor = loader.get_camera_sensor(camera_ids[0])
    detection_count = 0
    frame_stats: list[dict[str, Any]] = []
    for primary_index in selected_frame_indices(primary_sensor, args.frame_step, args.max_frames):
        timestamp_us = get_frame_timestamp_us(primary_sensor, primary_index)
        for camera_id in camera_ids:
            sensor = loader.get_camera_sensor(camera_id)
            frame_index = primary_index if camera_id == camera_ids[0] else closest_frame_index(sensor, timestamp_us)
            image = get_image_array(sensor, frame_index)
            detections = segmenter.detect(image)
            detection_count += len(detections)
            camera = extract_pinhole_camera(
                camera_id=camera_id, camera_model_id=1,
                model_parameters=sensor.model_parameters, image_shape=image.shape,
            )
            w2c = np.linalg.inv(get_sensor_c2w(sensor, frame_index))
            centers = [row["center_world"] for row in moving_cuboids if abs(int(row["timestamp_us"]) - int(get_frame_timestamp_us(sensor, frame_index))) <= 100_000]
            dynamic = np.zeros(image.shape[:2], dtype=bool)
            for center in centers:
                point = np.asarray([*center, 1.0], dtype=np.float64)
                cam = w2c @ point
                if cam[2] <= 1.0e-6:
                    continue
                u = int(round(camera.fx * cam[0] / cam[2] + camera.cx))
                v = int(round(camera.fy * cam[1] / cam[2] + camera.cy))
                if not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]):
                    continue
                for _class_id, _score, candidate in detections:
                    if candidate[v, u]:
                        dynamic |= candidate
            dynamic = dilate_dynamic_mask(dynamic, args.dilation_pixels)
            image_name = f"{camera_id}/{camera_id}_{primary_index:010d}_{frame_index:010d}.png"
            write_static_mask(mask_root, image_name, ~dynamic)
            write_overlay(mask_root, image_name, image, dynamic)
            frame_stats.append({"camera_id": camera_id, "primary_frame_index": int(primary_index), "frame_index": int(frame_index), "dynamic_pixels": int(dynamic.sum()), "static_pixels": int((~dynamic).sum())})
    return {
        "dataset": "nvidia_ncore", "scene_id": args.scene_id, "motion_source": "ncore_cuboid_tracks",
        "ncore_image_mask_matching": "projected_confirmed_cuboid_centroid_in_instance_mask", "segmentation_candidate_count": detection_count,
        "tracks": track_rows, "moving_cuboids": moving_cuboids, "frame_stats": frame_stats,
    }


def main() -> None:
    args = parse_args()
    mask_root = args.mask_root or Path("data/dynamic_masks") / args.dataset_name / args.scene_id
    segmenter = TorchvisionMaskRCNNSegmenter(args.weights_path, device=args.device, score_threshold=args.score_threshold)
    if args.dataset_name == "kitti360":
        payload = build_kitti(args, segmenter, mask_root)
    else:
        payload = build_ncore(args, segmenter, mask_root)
    payload.update({
        "weights_path": str(args.weights_path), "weights_sha256": segmenter.weights_sha256,
        "score_threshold": args.score_threshold, "dilation_pixels": args.dilation_pixels,
        "thresholds": {"min_observations": args.min_observations, "min_span_s": args.min_span_s, "min_displacement_m": args.min_displacement_m, "min_speed_mps": args.min_speed_mps},
    })
    path = write_manifest(mask_root, payload)
    print(f"Wrote dynamic mask artifact: {path}")


if __name__ == "__main__":
    main()
