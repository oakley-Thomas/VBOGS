#!/usr/bin/env python3

"""ROS2 node for online VBOGS uncertainty updates and NBV scoring."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_nbv import ScalarCam, load_octree_scene
from scripts.stereo_to_pointcloud import (
    StereoCalibration,
    build_matcher,
    build_validity_mask,
    unproject_to_world,
)
from vbogs.online import (
    atomic_save_npz,
    bucket_points_with_cache,
    load_anchor_grid_cache,
    normalize_online_observations,
    score_uncertainty_alpha,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/online/ros2_default.yaml"),
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_norm_params(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        "offset": np.asarray(raw["offset"], dtype=np.float32),
        "stdevs": np.asarray(raw["stdevs"], dtype=np.float32),
    }


def pose_to_c2w(position: Any, orientation: Any) -> np.ndarray:
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot
    c2w[:3, 3] = [position.x, position.y, position.z]
    return c2w


def image_msg_to_array(msg: Any) -> np.ndarray:
    encoding = msg.encoding.lower()
    channels = 1
    dtype = np.uint8
    if encoding in {"rgb8", "bgr8"}:
        channels = 3
    elif encoding in {"mono8", "8uc1"}:
        channels = 1
    elif encoding in {"32fc1"}:
        dtype = np.float32
        channels = 1
    else:
        raise ValueError(f"Unsupported ROS image encoding: {msg.encoding}")

    arr = np.frombuffer(msg.data, dtype=dtype)
    if channels == 1:
        arr = arr.reshape(int(msg.height), int(msg.width))
    else:
        arr = arr.reshape(int(msg.height), int(msg.width), channels)
        if encoding == "bgr8":
            arr = arr[..., ::-1]
    return np.ascontiguousarray(arr)


def make_stereo_args(config: dict[str, Any]) -> SimpleNamespace:
    stereo = config["stereo"]
    return SimpleNamespace(
        matcher=stereo.get("matcher", "sgbm"),
        matcher_min_disparity=int(stereo.get("matcher_min_disparity", 0)),
        num_disparities=int(stereo.get("num_disparities", 128)),
        block_size=int(stereo.get("block_size", 5)),
        uniqueness_ratio=int(stereo.get("uniqueness_ratio", 10)),
        speckle_window_size=int(stereo.get("speckle_window_size", 100)),
        speckle_range=int(stereo.get("speckle_range", 2)),
        disp12_max_diff=int(stereo.get("disp12_max_diff", 1)),
        min_disparity=float(stereo.get("min_disparity", 1.0)),
        lr_consistency_threshold=float(stereo.get("lr_consistency_threshold", 1.5)),
        texture_window_size=int(stereo.get("texture_window_size", 5)),
        texture_threshold=float(stereo.get("texture_threshold", 3.0)),
        pixel_step=int(stereo.get("pixel_step", 2)),
    )


def gray_for_stereo(rgb_or_gray: np.ndarray) -> np.ndarray:
    if rgb_or_gray.ndim == 2:
        return rgb_or_gray
    import cv2

    return cv2.cvtColor(rgb_or_gray, cv2.COLOR_RGB2GRAY)


class OnlineNbvNode:
    def __init__(self, config: dict[str, Any]) -> None:
        import rclpy
        from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
        from geometry_msgs.msg import PoseArray, PoseStamped
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String

        self.rclpy = rclpy
        self.msg_types = SimpleNamespace(
            CameraInfo=CameraInfo,
            DiagnosticArray=DiagnosticArray,
            DiagnosticStatus=DiagnosticStatus,
            Image=Image,
            KeyValue=KeyValue,
            PoseArray=PoseArray,
            PoseStamped=PoseStamped,
            String=String,
        )
        self.node = rclpy.create_node("vbogs_online_nbv")
        self.config = config
        self.online_cfg = config["online"]
        self.state_root = Path(self.online_cfg["state_root"]).resolve()
        self.batch_dir = self.state_root / config["handoff"].get("batch_dir", "batches")
        self.update_dir = self.state_root / config["handoff"].get("update_dir", "updates")
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.update_dir.mkdir(parents=True, exist_ok=True)
        self.cache = load_anchor_grid_cache(self.state_root / "anchor_grid_cache.npz")
        self.norm_params = load_norm_params(self.state_root / "norm_params.json")
        self.stereo_args = make_stereo_args(config)
        self.matcher = build_matcher(self.stereo_args)
        self.rng = np.random.default_rng(int(config["stereo"].get("random_seed", 0)))

        model_path_raw = self.online_cfg.get("model_path")
        if model_path_raw is None:
            manifest_path = self.state_root / "online_manifest.json"
            with manifest_path.open("r", encoding="utf-8") as handle:
                model_path_raw = json.load(handle).get("model_path")
        if model_path_raw is None:
            raise ValueError("online.model_path must be set, or online_manifest.json must contain model_path")
        model_path = Path(model_path_raw).resolve()
        scene, gaussians, _pipe = load_octree_scene(
            model_path,
            int(self.online_cfg.get("iteration", -1)),
            Path(self.online_cfg.get("octree_root", "Octree-AnyGS")),
            quiet=True,
        )
        self.scene = scene
        self.gaussians = gaussians
        self.device = self.online_cfg.get("device", "cuda")
        self.per_anchor_scalar = self.load_uncertainty_tensor()

        self.latest_left = None
        self.latest_right = None
        self.latest_camera_info = None
        self.latest_pose = None
        self.latest_candidates = None
        self.seq = 0
        self.last_applied_update_seq = -1

        topics = config["topics"]
        self.node.create_subscription(Image, topics["left_image"], self.on_left, 10)
        self.node.create_subscription(Image, topics["right_image"], self.on_right, 10)
        self.node.create_subscription(CameraInfo, topics["left_camera_info"], self.on_camera_info, 10)
        self.node.create_subscription(PoseStamped, topics["current_pose"], self.on_pose, 10)
        self.node.create_subscription(PoseArray, topics["candidate_poses"], self.on_candidates, 10)
        self.best_pose_pub = self.node.create_publisher(PoseStamped, topics["best_pose"], 10)
        self.score_pub = self.node.create_publisher(String, topics["candidate_scores"], 10)
        self.diag_pub = self.node.create_publisher(DiagnosticArray, topics["diagnostics"], 10)
        self.unc_pub = None
        if bool(self.online_cfg.get("publish_uncertainty_image", False)):
            self.unc_pub = self.node.create_publisher(Image, topics["uncertainty_image"], 10)
        self.node.create_timer(float(config["handoff"].get("poll_updates_sec", 0.05)), self.tick)

    def load_uncertainty_tensor(self):
        import torch

        values = np.asarray(np.load(self.state_root / "U_online.npy"), dtype=np.float32)
        return torch.from_numpy(values).to(device=self.device, dtype=torch.float32)

    def on_left(self, msg: Any) -> None:
        self.latest_left = msg

    def on_right(self, msg: Any) -> None:
        self.latest_right = msg

    def on_camera_info(self, msg: Any) -> None:
        self.latest_camera_info = msg

    def on_pose(self, msg: Any) -> None:
        self.latest_pose = msg

    def on_candidates(self, msg: Any) -> None:
        self.latest_candidates = msg

    def poll_updates(self) -> bool:
        updates = sorted(self.update_dir.glob("*.npz"))
        if not updates:
            return False
        latest = updates[-1]
        seq = int(latest.stem)
        if seq <= self.last_applied_update_seq:
            return False
        self.per_anchor_scalar = self.load_uncertainty_tensor()
        self.last_applied_update_seq = seq
        return True

    def camera_info_calibration(self) -> StereoCalibration:
        info = self.latest_camera_info
        k = info.k
        return StereoCalibration(
            width=int(info.width),
            height=int(info.height),
            fx=float(k[0]),
            fy=float(k[4]),
            cx=float(k[2]),
            cy=float(k[5]),
            baseline_m=float(self.config["stereo"]["baseline_m"]),
        )

    def write_current_batch(self, timings: dict[str, float]) -> dict[str, Any]:
        left_rgb = image_msg_to_array(self.latest_left)
        right_rgb = image_msg_to_array(self.latest_right)
        left_gray = gray_for_stereo(left_rgb)
        right_gray = gray_for_stereo(right_rgb)
        c2w = pose_to_c2w(self.latest_pose.pose.position, self.latest_pose.pose.orientation)
        calibration = self.camera_info_calibration()

        t0 = time.perf_counter()
        stereo = self.matcher.compute(left_gray, right_gray)
        valid = build_validity_mask(stereo.disparity_left, stereo.disparity_right, left_gray, self.stereo_args)
        xyz_world, rgb = unproject_to_world(
            disparity_left=stereo.disparity_left,
            rgb_image=left_rgb if left_rgb.ndim == 3 else np.repeat(left_rgb[..., None], 3, axis=2),
            calibration=calibration,
            c2w=c2w,
            valid_mask=valid,
            max_depth_m=float(self.config["stereo"].get("max_depth_m", 120.0)),
            max_points_per_frame=int(self.config["stereo"].get("max_points_per_frame", 150000)),
            rng=self.rng,
        )
        timings["stereo_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        points_norm, norm_stats = normalize_online_observations(
            xyz_world,
            rgb,
            self.norm_params,
            outlier_z=float(self.config["normalization"].get("outlier_z", 6.0)),
        )
        timings["normalization_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        bucketed = bucket_points_with_cache(xyz_world, self.cache)
        timings["bucketing_ms"] = (time.perf_counter() - t0) * 1000.0

        batch_path = self.batch_dir / f"{self.seq:010d}.npz"
        atomic_save_npz(
            batch_path,
            seq=np.array(self.seq, dtype=np.int64),
            stamp_sec=np.array(self.node.get_clock().now().nanoseconds / 1.0e9, dtype=np.float64),
            points_norm=points_norm,
            anchor_offsets=bucketed["anchor_offsets"],
            point_indices=bucketed["point_indices"],
            point_counts=bucketed["point_counts"],
            touched_anchor_ids=bucketed["touched_anchor_ids"],
            normalization_outlier_count=np.array(norm_stats["outlier_count"], dtype=np.int32),
        )
        result = {
            "batch_path": str(batch_path),
            "point_count": int(points_norm.shape[0]),
            "touched_anchor_count": int(bucketed["touched_anchor_ids"].shape[0]),
            "normalization": norm_stats,
        }
        self.seq += 1
        return result

    def make_candidate_cam(self, pose: Any, index: int) -> ScalarCam:
        info = self.latest_camera_info
        c2w = pose_to_c2w(pose.position, pose.orientation)
        reference = SimpleNamespace(
            uid=index,
            image_name=f"planner_{index:04d}",
            image_path="",
            resolution_scale=1.0,
            image_width=int(info.width),
            image_height=int(info.height),
            FoVx=2.0 * math.atan(float(info.width) / (2.0 * float(info.k[0]))),
            FoVy=2.0 * math.atan(float(info.height) / (2.0 * float(info.k[4]))),
            fx=float(info.k[0]),
            fy=float(info.k[4]),
            cx=float(info.k[2]),
            cy=float(info.k[5]),
            znear=0.01,
            zfar=100.0,
        )
        return ScalarCam(reference, c2w, index, f"planner_{index:04d}")

    def score_candidates(self, timings: dict[str, float]) -> dict[str, Any] | None:
        import torch
        from vbogs.render import render_scalar

        if self.latest_candidates is None or not self.latest_candidates.poses:
            return None
        max_candidates = int(self.online_cfg.get("max_candidates", 32))
        candidates = self.latest_candidates.poses[:max_candidates]
        rows = []
        best = None
        t0 = time.perf_counter()
        with torch.no_grad():
            for idx, pose in enumerate(candidates):
                cam = self.make_candidate_cam(pose, idx)
                rendered = render_scalar(
                    cam,
                    self.gaussians,
                    self.per_anchor_scalar,
                    iteration=2_147_483_647,
                    force_all_levels=bool(self.online_cfg.get("force_all_levels", False)),
                )
                score, unc_sum, alpha_sum = score_uncertainty_alpha(
                    rendered["unc_image"].detach().cpu().numpy(),
                    rendered["alpha_image"].detach().cpu().numpy(),
                )
                row = {
                    "candidate_index": idx,
                    "score": score,
                    "unc_sum": unc_sum,
                    "alpha_sum": alpha_sum,
                }
                rows.append(row)
                if best is None or score > best["score"]:
                    best = row
        timings["render_scoring_ms"] = (time.perf_counter() - t0) * 1000.0
        if best is None:
            return None
        return {
            "best": best,
            "rows": sorted(rows, key=lambda row: row["score"], reverse=True),
            "best_pose": candidates[int(best["candidate_index"])],
        }

    def publish_results(self, batch_info: dict[str, Any], scoring: dict[str, Any] | None, timings: dict[str, float]) -> None:
        msgs = self.msg_types
        stale_update = (self.seq - 1) > self.last_applied_update_seq
        if scoring is not None:
            pose_msg = msgs.PoseStamped()
            pose_msg.header = self.latest_candidates.header
            pose_msg.pose = scoring["best_pose"]
            self.best_pose_pub.publish(pose_msg)
            score_msg = msgs.String()
            score_msg.data = json.dumps(
                {
                    "seq": self.seq - 1,
                    "stale_update": stale_update,
                    "best": scoring["best"],
                    "top_k": scoring["rows"][:10],
                },
                sort_keys=True,
            )
            self.score_pub.publish(score_msg)

        status = msgs.DiagnosticStatus()
        status.level = 1 if stale_update else 0
        status.name = "vbogs_online_nbv"
        status.message = "stale uncertainty update" if stale_update else "ok"
        values = {
            "seq": self.seq - 1,
            "last_applied_update_seq": self.last_applied_update_seq,
            "point_count": batch_info["point_count"],
            "touched_anchor_count": batch_info["touched_anchor_count"],
            **timings,
        }
        status.values = [msgs.KeyValue(key=str(key), value=str(value)) for key, value in values.items()]
        diag = msgs.DiagnosticArray()
        diag.status = [status]
        self.diag_pub.publish(diag)

    def tick(self) -> None:
        self.poll_updates()
        if (
            self.latest_left is None
            or self.latest_right is None
            or self.latest_camera_info is None
            or self.latest_pose is None
        ):
            return
        start = time.perf_counter()
        timings: dict[str, float] = {}
        try:
            batch_info = self.write_current_batch(timings)
            scoring = self.score_candidates(timings)
            timings["total_ms"] = (time.perf_counter() - start) * 1000.0
            self.publish_results(batch_info, scoring, timings)
        except Exception as exc:  # pragma: no cover - live ROS guardrail
            self.node.get_logger().error(f"VBOGS online tick failed: {exc}")


def main() -> None:
    args = parse_args()
    try:
        import rclpy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("ROS2 Python packages are required to run this node: install ROS2 Humble.") from exc

    config = load_config(args.config)
    rclpy.init()
    wrapper = OnlineNbvNode(config)
    try:
        rclpy.spin(wrapper.node)
    finally:
        wrapper.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
