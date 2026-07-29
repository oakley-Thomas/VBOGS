#!/usr/bin/env python3

"""Rank NVIDIA PhysicalAI AV NCore clips by their suitability for neural reconstruction.

Static-scene Gaussian-splatting reconstruction quality is driven by a handful of
clip properties that are all measurable before training:

- **Static scene**: moving actors violate the static-world assumption and leave
  streaks/floaters. Scored from autolabelled cuboid tracks, weighted by how close
  the moving actor passes to the ego rig.
- **Parallax**: triangulation needs camera baseline that is large relative to the
  scene depth. Scored from ego path length over median LiDAR range.
- **View diversity**: a clip driven in a straight line constrains distant geometry
  poorly. Scored from the ego yaw sweep.
- **Photometric quality**: night, over/under exposure, and motion blur all cap the
  achievable PSNR. Scored from sampled primary-camera frames.
- **Geometry seed**: LiDAR return density determines how well the sparse seed
  covers the scene.

Each sub-score is normalized to 0-1 and combined into a weighted total, so clips
can be ranked and the weak axis of any clip can be read off directly.

Examples
--------
Score every downloaded clip and print the ranking table:

    python scripts/score_ncore_clips_for_reconstruction.py

Score specific clips and emit JSON for downstream tooling:

    python scripts/score_ncore_clips_for_reconstruction.py \\
        --scene-id 000da9de-0ee5-465a-9a2d-e7e91d3016bb \\
        --scene-id 004c2001-5fc3-43b1-a4d8-bfb0bbb9fdc6 \\
        --json

Requires the `nvidia-ncore` package, so run it inside `vbogs-torch`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.data_layout import resolve_nvidia_ncore_root
from vbogs.ncore_adapter import (
    DEFAULT_CAMERA_IDS,
    DEFAULT_LIDAR_ID,
    get_frame_count,
    get_frame_timestamp_us,
    get_image_array,
    load_ncore_sequence_loader,
)

# A track is "moving" once its world-frame centroid travels past this distance.
# Autolabel jitter on a parked car stays well under half a metre.
MOVING_TRACK_MIN_DISPLACEMENT_M = 1.0

# Moving actors inside this radius dominate the primary camera's field of view,
# so they are the ones that actually corrupt the reconstruction.
NEARBY_ACTOR_RADIUS_M = 30.0

# Below this the rig is effectively parked and consecutive frames are redundant.
STATIONARY_SPEED_MPS = 0.5

# Top fraction of the frame treated as sky for the daylight test. Validated on 14
# hand-labelled clips: night sky value <= 68, day/dusk >= 108, with no overlap.
SKY_BAND_FRACTION = 0.25

# Sub-score weights. Static scene and parallax dominate because no amount of
# training recovers from a dynamic scene or a degenerate baseline.
SCORE_WEIGHTS = {
    "static": 0.30,
    "parallax": 0.25,
    "photometric": 0.25,
    "view_diversity": 0.10,
    "geometry_seed": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scene-id",
        action="append",
        default=None,
        help="Clip id to score. Repeatable. Defaults to every clip under the NCore root.",
    )
    parser.add_argument("--ncore-root", type=Path, default=None)
    parser.add_argument(
        "--camera-id",
        default=DEFAULT_CAMERA_IDS[0],
        help="Primary camera used for pose and photometric sampling.",
    )
    parser.add_argument("--lidar-id", default=DEFAULT_LIDAR_ID)
    parser.add_argument(
        "--pose-frame-step",
        type=int,
        default=5,
        help="Frame step for the ego-motion trajectory scan.",
    )
    parser.add_argument(
        "--image-samples",
        type=int,
        default=12,
        help="Number of primary-camera frames sampled for the photometric score.",
    )
    parser.add_argument(
        "--lidar-samples",
        type=int,
        default=8,
        help="Number of LiDAR frames sampled for range and density stats.",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help=(
            "Score from the core NCore file alone (ego motion + cuboids), skipping "
            "imagery and LiDAR. Works on `--mode core-only` downloads, which are ~9 MB "
            "per clip instead of ~2 GB, so the whole dataset can be triaged cheaply. "
            "Applied automatically when a clip has no camera component."
        ),
    )
    parser.add_argument(
        "--ignore-lidar",
        action="store_true",
        help=(
            "Skip LiDAR scoring even when the component is present, so clips downloaded "
            "at different tiers are ranked on the same axes."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Only report the N best clips. 0 reports all.",
    )
    return parser.parse_args()


def discover_scene_ids(ncore_root: Path) -> list[str]:
    if not ncore_root.is_dir():
        raise SystemExit(f"NCore root not found: {ncore_root}")
    return sorted(entry.name for entry in ncore_root.iterdir() if entry.is_dir())


def unit_clamp(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def plateau_score(value: float, low: float, ideal_low: float, ideal_high: float, high: float) -> float:
    """Ramp up to 1.0 across [low, ideal_low], hold, then ramp down to 0 at `high`."""
    if value <= low or value >= high:
        return 0.0
    if value < ideal_low:
        return unit_clamp((value - low) / (ideal_low - low))
    if value <= ideal_high:
        return 1.0
    return unit_clamp((high - value) / (high - ideal_high))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rig_to_world(loader: Any, timestamps_us: np.ndarray) -> np.ndarray:
    """Interpolated rig-to-world poses, as an (N, 4, 4) stack.

    The pose graph evaluates at arbitrary timestamps, so cuboid observations are
    lifted at their own capture time rather than snapped to a camera frame.
    """
    timestamps = np.asarray(timestamps_us, dtype=np.uint64)
    poses = np.asarray(loader.pose_graph.evaluate_poses("rig", "world", timestamps), dtype=np.float64)
    return poses.reshape(-1, 4, 4)


def screening_timestamps_us(loader: Any, rate_hz: float = 10.0) -> np.ndarray:
    """Uniform timestamps across the clip, for core-only clips with no camera.

    A `--mode core-only` download has no camera component, so frame timestamps
    have to come from the sequence interval instead. The pose graph interpolates,
    so any sampling rate gives the same trajectory.
    """
    interval = loader.sequence_timestamp_interval_us
    if hasattr(interval, "start"):
        start, stop = int(interval.start), int(interval.stop)
    else:
        start, stop = int(interval[0]), int(interval[1])
    count = max(2, int((stop - start) * 1e-6 * rate_hz))
    # The interval is half-closed, so stop is exclusive; back off a microsecond
    # rather than dropping a whole sample interval off the end of the trajectory.
    return np.linspace(start, stop - 1, num=count, endpoint=True, dtype=np.uint64)


def collect_ego_motion(loader: Any, sensor: Any, frame_step: int) -> dict[str, Any]:
    if sensor is None:
        timestamps_us = screening_timestamps_us(loader)
        frame_count = 0
    else:
        frame_count = get_frame_count(sensor)
        indices = list(range(frame_count))[:: max(1, frame_step)]
        timestamps_us = np.asarray(
            [get_frame_timestamp_us(sensor, index) for index in indices], dtype=np.uint64
        )

    poses = rig_to_world(loader, timestamps_us)
    positions = poses[:, :3, 3]
    # Rig convention is x-forward, so the first column is the heading vector.
    headings = poses[:, :3, 0]
    yaws = np.arctan2(headings[:, 1], headings[:, 0])
    timestamps = timestamps_us.astype(np.float64) * 1e-6
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dt = np.diff(timestamps)
    speeds = np.divide(steps, dt, out=np.zeros_like(steps), where=dt > 0)

    yaw_deltas = np.asarray([abs(wrap_angle(yaws[i + 1] - yaws[i])) for i in range(len(yaws) - 1)])

    return {
        "frame_count": frame_count,
        "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "path_length_m": float(steps.sum()),
        "net_displacement_m": float(np.linalg.norm(positions[-1] - positions[0])) if len(positions) > 1 else 0.0,
        "median_speed_mps": float(np.median(speeds)) if speeds.size else 0.0,
        "stationary_frac": float(np.mean(speeds < STATIONARY_SPEED_MPS)) if speeds.size else 1.0,
        "yaw_sweep_deg": float(np.degrees(yaw_deltas.sum())) if yaw_deltas.size else 0.0,
        "positions": positions,
        "timestamps_s": timestamps,
    }


def collect_dynamic_actors(loader: Any) -> dict[str, Any]:
    """Lift rig-frame cuboid tracks into world coordinates and measure real motion.

    Cuboids are stored relative to the moving rig, so a parked car appears to
    translate. Composing with the ego pose at each observation's own timestamp is
    what separates genuinely moving actors from the ego's motion; without it every
    track looks dynamic.
    """
    observations = list(loader.get_cuboid_track_observations())
    if not observations:
        return {
            "total_tracks": 0,
            "moving_tracks": 0,
            "nearby_moving_tracks": 0,
            "moving_frac": 0.0,
            "moving_classes": {},
        }

    timestamps_us = np.asarray([obs.timestamp_us for obs in observations], dtype=np.uint64)
    centroids_rig = np.asarray([obs.bbox3.centroid for obs in observations], dtype=np.float64)
    poses = rig_to_world(loader, timestamps_us)
    centroids_world = np.einsum("nij,nj->ni", poses[:, :3, :3], centroids_rig) + poses[:, :3, 3]
    ranges_m = np.linalg.norm(centroids_rig, axis=1)

    tracks: dict[str, list[tuple[np.ndarray, float]]] = defaultdict(list)
    class_ids: dict[str, str] = {}
    for observation, centroid_world, range_m in zip(observations, centroids_world, ranges_m):
        tracks[observation.track_id].append((centroid_world, float(range_m)))
        class_ids.setdefault(observation.track_id, str(observation.class_id))

    total_tracks = len(tracks)
    moving_tracks = 0
    nearby_moving_tracks = 0
    moving_classes: dict[str, int] = defaultdict(int)

    for track_id, observations in tracks.items():
        if len(observations) < 2:
            continue
        points = np.asarray([point for point, _ in observations])
        displacement = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        if displacement < MOVING_TRACK_MIN_DISPLACEMENT_M:
            continue
        moving_tracks += 1
        moving_classes[class_ids.get(track_id, "unknown")] += 1
        if min(range_m for _, range_m in observations) <= NEARBY_ACTOR_RADIUS_M:
            nearby_moving_tracks += 1

    return {
        "total_tracks": total_tracks,
        "moving_tracks": moving_tracks,
        "nearby_moving_tracks": nearby_moving_tracks,
        "moving_frac": float(moving_tracks / total_tracks) if total_tracks else 0.0,
        "moving_classes": dict(sorted(moving_classes.items(), key=lambda kv: -kv[1])),
    }


def laplacian_variance(gray: np.ndarray) -> float:
    """Blur proxy: high-frequency energy via a 4-neighbour Laplacian."""
    center = gray[1:-1, 1:-1]
    lap = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
    )
    return float(lap.var())


def collect_photometrics(sensor: Any, sample_count: int) -> dict[str, Any]:
    frame_count = get_frame_count(sensor)
    indices = np.linspace(0, frame_count - 1, num=min(sample_count, frame_count), dtype=int)

    brightness = []
    sky_luma = []
    sharpness = []
    clipped = []
    for index in indices:
        image = get_image_array(sensor, int(index))
        gray = image.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
        brightness.append(gray.mean())
        # Sky band, measured as max-channel value rather than luma. Two traps:
        # whole-frame brightness cannot tell a night snowstorm from an overcast day
        # (snow reflects streetlight into the mid-tones), and luma weights blue at
        # 0.114, so a clear blue sky reads darker than an overcast white one and
        # sunny clips get penalised. Max-channel avoids both.
        band = image[: int(image.shape[0] * SKY_BAND_FRACTION)]
        sky_luma.append(float(band.max(axis=2).mean()))
        sharpness.append(laplacian_variance(gray))
        clipped.append(float(np.mean((gray < 5.0) | (gray > 250.0))))

    return {
        "brightness_mean": float(np.mean(brightness)),
        "brightness_min": float(np.min(brightness)),
        "sky_luma_mean": float(np.mean(sky_luma)),
        "sharpness_mean": float(np.mean(sharpness)),
        "sharpness_min": float(np.min(sharpness)),
        "clipped_frac": float(np.mean(clipped)),
        "image_height": int(get_image_array(sensor, int(indices[0])).shape[0]),
        "image_width": int(get_image_array(sensor, int(indices[0])).shape[1]),
    }


def collect_lidar_stats(loader: Any, lidar_id: str, sample_count: int) -> dict[str, Any]:
    from scripts.prepare_nvidia_ncore_colmap import _point_cloud_xyz

    sensor = loader.get_lidar_sensor(lidar_id)
    frame_count = get_frame_count(sensor)
    indices = np.linspace(0, frame_count - 1, num=min(sample_count, frame_count), dtype=int)

    counts = []
    ranges = []
    for index in indices:
        try:
            cloud = sensor.get_frame_point_cloud(
                int(index), motion_compensation=True, with_start_points=False
            )
        except TypeError:
            cloud = sensor.get_frame_point_cloud(int(index))
        xyz = np.asarray(_point_cloud_xyz(cloud), dtype=np.float64).reshape(-1, 3)
        counts.append(xyz.shape[0])
        if xyz.size:
            ranges.append(float(np.median(np.linalg.norm(xyz, axis=1))))

    return {
        "lidar_frame_count": frame_count,
        "points_per_frame": float(np.mean(counts)) if counts else 0.0,
        "median_range_m": float(np.median(ranges)) if ranges else float("nan"),
    }


def score_clip(metrics: dict[str, Any]) -> dict[str, float]:
    ego = metrics["ego_motion"]
    actors = metrics["dynamic_actors"]
    photo = metrics.get("photometrics")
    lidar = metrics.get("lidar")

    # Static scene: nearby movers are what actually break the reconstruction, so
    # they carry twice the penalty of a distant one. Decay exponentially rather
    # than linearly - busy urban clips would all clamp to zero on a linear ramp
    # and stop ranking against each other.
    dynamic_load = actors["nearby_moving_tracks"] + 0.5 * (
        actors["moving_tracks"] - actors["nearby_moving_tracks"]
    )
    static = float(math.exp(-dynamic_load / 30.0))

    # Parallax: baseline must be meaningful relative to how far away the scene is.
    # Without LiDAR (core-only screening) fall back on absolute path length, which
    # is a coarser proxy because it cannot tell an open road from a tight street.
    if lidar is not None and math.isfinite(lidar["median_range_m"]) and lidar["median_range_m"] > 0:
        parallax_ratio = ego["path_length_m"] / lidar["median_range_m"]
        parallax = plateau_score(parallax_ratio, low=0.0, ideal_low=4.0, ideal_high=30.0, high=120.0)
    else:
        parallax = plateau_score(
            ego["path_length_m"], low=0.0, ideal_low=75.0, ideal_high=600.0, high=2000.0
        )
    # A rig parked for most of the clip yields redundant views regardless of ratio.
    parallax *= unit_clamp(1.0 - ego["stationary_frac"])

    # View diversity: turns reveal occluded structure; a straight run does not.
    view_diversity = unit_clamp(ego["yaw_sweep_deg"] / 90.0)

    parts = {
        "static": static,
        "parallax": parallax,
        "view_diversity": view_diversity,
    }

    if photo is not None:
        # Photometric: daylight, unclipped, and sharp. Daylight is judged from the
        # sky band rather than frame brightness - see collect_photometrics.
        daylight = plateau_score(
            photo["sky_luma_mean"], low=55.0, ideal_low=105.0, ideal_high=235.0, high=254.0
        )
        blur = unit_clamp(photo["sharpness_min"] / 150.0)
        clipping = unit_clamp(1.0 - photo["clipped_frac"] / 0.15)
        parts["photometric"] = 0.45 * daylight + 0.35 * blur + 0.20 * clipping

    if lidar is not None:
        parts["geometry_seed"] = unit_clamp(lidar["points_per_frame"] / 150_000.0)

    # Renormalize over whatever axes were measurable so screening totals stay on
    # the same 0-1 scale as full scores, rather than being capped by missing axes.
    weight_total = sum(SCORE_WEIGHTS[key] for key in parts)
    parts["total"] = float(
        sum(SCORE_WEIGHTS[key] * value for key, value in parts.items()) / weight_total
    )
    return {key: round(value, 4) for key, value in parts.items()}


def evaluate_clip(args: argparse.Namespace, ncore_root: Path, scene_id: str) -> dict[str, Any]:
    loader = load_ncore_sequence_loader(ncore_root, scene_id)
    # A core-only download exposes no sensors, so screening mode is also the
    # automatic fallback when the requested camera is not present.
    screening = args.screen or args.camera_id not in set(loader.camera_ids)
    sensor = None if screening else loader.get_camera_sensor(args.camera_id)

    metrics: dict[str, Any] = {
        "scene_id": scene_id,
        "camera_id": None if screening else args.camera_id,
        "screening": screening,
        "ego_motion": collect_ego_motion(loader, sensor, args.pose_frame_step),
        "dynamic_actors": collect_dynamic_actors(loader),
    }
    if not screening:
        metrics["photometrics"] = collect_photometrics(sensor, args.image_samples)
        # The triage tier fetches the camera component only, so LiDAR is scored
        # when present rather than assumed. `--ignore-lidar` forces it off so a
        # mixed pool of camera-only and full clips is ranked on identical axes.
        if not args.ignore_lidar and args.lidar_id in set(loader.lidar_ids):
            metrics["lidar"] = collect_lidar_stats(loader, args.lidar_id, args.lidar_samples)
    # Trajectory arrays are only needed for the derived statistics above.
    metrics["ego_motion"].pop("positions", None)
    metrics["ego_motion"].pop("timestamps_s", None)
    metrics["scores"] = score_clip(metrics)
    return metrics


def format_table(results: Sequence[dict[str, Any]]) -> str:
    header = (
        f"{'rank':<5}{'scene_id':<40}{'total':>7}{'static':>8}{'parlx':>7}"
        f"{'photo':>7}{'view':>7}{'seed':>7}  {'notes'}"
    )
    lines = [header, "-" * len(header)]
    for rank, result in enumerate(results, start=1):
        scores = result["scores"]
        ego = result["ego_motion"]
        actors = result["dynamic_actors"]
        photo = result.get("photometrics")
        notes = (
            f"path={ego['path_length_m']:.0f}m yaw={ego['yaw_sweep_deg']:.0f}deg "
            f"movers={actors['moving_tracks']}({actors['nearby_moving_tracks']} near)"
        )
        notes += f" lum={photo['brightness_mean']:.0f}" if photo else " [screened]"

        def cell(name: str) -> str:
            return f"{scores[name]:>7.3f}" if name in scores else f"{'-':>7}"

        lines.append(
            f"{rank:<5}{result['scene_id']:<40}{scores['total']:>7.3f}{scores['static']:>8.3f}"
            f"{cell('parallax')}{cell('photometric')}"
            f"{cell('view_diversity')}{cell('geometry_seed')}  {notes}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ncore_root = resolve_nvidia_ncore_root(args.ncore_root)
    scene_ids = args.scene_id or discover_scene_ids(ncore_root)

    results = []
    for scene_id in scene_ids:
        try:
            results.append(evaluate_clip(args, ncore_root, scene_id))
        except Exception as error:  # noqa: BLE001 - one bad clip must not sink the sweep
            print(f"[skip] {scene_id}: {error}", file=sys.stderr)

    results.sort(key=lambda result: result["scores"]["total"], reverse=True)
    if args.top:
        results = results[: args.top]

    if args.json:
        print(json.dumps({"ncore_root": str(ncore_root), "clips": results}, indent=2))
    else:
        print(format_table(results))


if __name__ == "__main__":
    main()
