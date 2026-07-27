import math

import numpy as np
import pytest

from scripts.score_ncore_clips_for_reconstruction import (
    collect_dynamic_actors,
    collect_ego_motion,
    plateau_score,
    score_clip,
    unit_clamp,
)


class FakeBBox3:
    def __init__(self, centroid):
        self.centroid = centroid


class FakeObservation:
    def __init__(self, track_id, class_id, timestamp_us, centroid):
        self.track_id = track_id
        self.class_id = class_id
        self.timestamp_us = timestamp_us
        self.bbox3 = FakeBBox3(centroid)


class FakePoseGraph:
    """Ego rig driving straight down +x at 10 m/s while yawing steadily."""

    def evaluate_poses(self, source_node, target_node, timestamps_us):
        assert (source_node, target_node) == ("rig", "world")
        poses = []
        for timestamp in np.asarray(timestamps_us, dtype=np.float64):
            seconds = timestamp * 1e-6
            yaw = math.radians(3.0) * seconds
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            pose = np.eye(4)
            pose[:3, :3] = np.array(
                [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]]
            )
            pose[:3, 3] = np.array([10.0 * seconds, 0.0, 0.0])
            poses.append(pose)
        return np.asarray(poses)


class FakeCameraSensor:
    def __init__(self, frames_count=11, step_us=100_000):
        self.frames_count = frames_count
        self._step_us = step_us

    def get_frame_timestamp_us(self, frame_index):
        return int(frame_index) * self._step_us


class FakeInterval:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop


class FakeLoader:
    def __init__(self, observations, interval_us=(0, 1_000_000)):
        self.pose_graph = FakePoseGraph()
        self._observations = observations
        self.sequence_timestamp_interval_us = FakeInterval(*interval_us)

    def get_cuboid_track_observations(self):
        return iter(self._observations)


def rig_centroid_for_static_world_point(world_xyz, timestamp_us):
    """Invert the fake ego pose so a world-fixed point becomes a rig observation."""
    pose = FakePoseGraph().evaluate_poses("rig", "world", np.asarray([timestamp_us]))[0]
    return tuple(np.linalg.inv(pose)[:3, :3] @ np.asarray(world_xyz) + np.linalg.inv(pose)[:3, 3])


def test_static_actor_is_not_counted_as_moving_under_ego_motion():
    # A parked car sits at a fixed world position; only the rig moves. Without
    # ego compensation its rig-frame centroid sweeps ~10 m and looks dynamic.
    observations = [
        FakeObservation(
            "parked",
            "automobile",
            timestamp,
            rig_centroid_for_static_world_point((25.0, 4.0, 0.0), timestamp),
        )
        for timestamp in (0, 250_000, 500_000, 750_000, 1_000_000)
    ]

    actors = collect_dynamic_actors(FakeLoader(observations))

    assert actors["total_tracks"] == 1
    assert actors["moving_tracks"] == 0
    assert actors["nearby_moving_tracks"] == 0


def test_moving_actor_is_detected_and_classified_by_proximity():
    # Crosses 12 m of world space right next to the rig.
    near_mover = [
        FakeObservation("near", "person", timestamp, (5.0, -4.0 + 12.0 * timestamp * 1e-6, 0.0))
        for timestamp in (0, 500_000, 1_000_000)
    ]
    # Same motion but far outside the nearby radius.
    far_mover = [
        FakeObservation("far", "automobile", timestamp, (90.0, 12.0 * timestamp * 1e-6, 0.0))
        for timestamp in (0, 500_000, 1_000_000)
    ]

    actors = collect_dynamic_actors(FakeLoader(near_mover + far_mover))

    assert actors["moving_tracks"] == 2
    assert actors["nearby_moving_tracks"] == 1
    assert actors["moving_classes"] == {"person": 1, "automobile": 1}


def test_collect_ego_motion_recovers_speed_and_yaw_sweep():
    ego = collect_ego_motion(FakeLoader([]), FakeCameraSensor(), frame_step=1)

    assert ego["frame_count"] == 11
    assert ego["path_length_m"] == pytest.approx(10.0, abs=0.05)
    assert ego["median_speed_mps"] == pytest.approx(10.0, abs=0.05)
    assert ego["stationary_frac"] == 0.0
    # 3 deg/s over the 1 s the fake sensor spans.
    assert ego["yaw_sweep_deg"] == pytest.approx(3.0, abs=0.05)


def test_unit_clamp_and_plateau_score_bounds():
    assert unit_clamp(-1.0) == 0.0
    assert unit_clamp(2.0) == 1.0

    assert plateau_score(0.0, 0.0, 4.0, 30.0, 120.0) == 0.0
    assert plateau_score(2.0, 0.0, 4.0, 30.0, 120.0) == 0.5
    assert plateau_score(10.0, 0.0, 4.0, 30.0, 120.0) == 1.0
    assert plateau_score(200.0, 0.0, 4.0, 30.0, 120.0) == 0.0


def build_metrics(*, moving, nearby, path_m, yaw_deg, brightness, sharpness, stationary=0.0):
    return {
        "ego_motion": {
            "path_length_m": path_m,
            "yaw_sweep_deg": yaw_deg,
            "stationary_frac": stationary,
        },
        "dynamic_actors": {"moving_tracks": moving, "nearby_moving_tracks": nearby},
        "photometrics": {
            "brightness_mean": brightness,
            "sharpness_min": sharpness,
            "clipped_frac": 0.0,
        },
        "lidar": {"points_per_frame": 300_000.0, "median_range_m": 12.0},
    }


def test_clean_clip_outscores_dynamic_night_clip():
    clean = score_clip(
        build_metrics(moving=4, nearby=1, path_m=150.0, yaw_deg=80.0, brightness=110.0, sharpness=140.0)
    )
    messy = score_clip(
        build_metrics(moving=130, nearby=80, path_m=150.0, yaw_deg=10.0, brightness=45.0, sharpness=10.0)
    )

    assert clean["total"] > messy["total"]
    assert clean["static"] > messy["static"]
    assert clean["view_diversity"] > messy["view_diversity"]
    assert clean["photometric"] > messy["photometric"]


def test_stationary_rig_loses_parallax_credit():
    moving_rig = score_clip(
        build_metrics(moving=0, nearby=0, path_m=100.0, yaw_deg=30.0, brightness=110.0, sharpness=140.0)
    )
    parked_rig = score_clip(
        build_metrics(
            moving=0, nearby=0, path_m=100.0, yaw_deg=30.0, brightness=110.0, sharpness=140.0,
            stationary=0.9,
        )
    )

    assert parked_rig["parallax"] < moving_rig["parallax"]


def test_screening_ego_motion_matches_camera_driven_ego_motion():
    """Core-only screening must recover the same trajectory as a full clip.

    The pose graph interpolates, so sampling from the sequence interval instead of
    camera frame timestamps should not change the measured motion.
    """
    loader = FakeLoader([])

    with_camera = collect_ego_motion(loader, FakeCameraSensor(), frame_step=1)
    screened = collect_ego_motion(loader, None, frame_step=1)

    assert screened["path_length_m"] == pytest.approx(with_camera["path_length_m"], rel=0.02)
    assert screened["yaw_sweep_deg"] == pytest.approx(with_camera["yaw_sweep_deg"], rel=0.02)
    assert screened["frame_count"] == 0


def test_screening_score_omits_unmeasurable_axes_and_stays_comparable():
    full_metrics = build_metrics(
        moving=4, nearby=1, path_m=150.0, yaw_deg=80.0, brightness=110.0, sharpness=140.0
    )
    screened_metrics = {
        "ego_motion": full_metrics["ego_motion"],
        "dynamic_actors": full_metrics["dynamic_actors"],
    }

    full = score_clip(full_metrics)
    screened = score_clip(screened_metrics)

    assert "photometric" not in screened
    assert "geometry_seed" not in screened
    assert screened["static"] == full["static"]
    # Renormalization keeps the screened total on the same 0-1 scale rather than
    # capping it at the sum of the surviving weights (0.65).
    assert screened["total"] > 0.65


def test_screening_preserves_ranking_of_clearly_different_clips():
    def screened(moving, nearby, path_m, yaw_deg):
        metrics = build_metrics(
            moving=moving, nearby=nearby, path_m=path_m, yaw_deg=yaw_deg,
            brightness=110.0, sharpness=140.0,
        )
        return score_clip(
            {"ego_motion": metrics["ego_motion"], "dynamic_actors": metrics["dynamic_actors"]}
        )

    good = screened(moving=14, nearby=9, path_m=173.0, yaw_deg=88.0)
    poor = screened(moving=129, nearby=78, path_m=197.0, yaw_deg=14.0)

    assert good["total"] > poor["total"]


def test_scores_stay_within_unit_range():
    scores = score_clip(
        build_metrics(moving=0, nearby=0, path_m=1e6, yaw_deg=1e4, brightness=128.0, sharpness=1e5)
    )
    for name, value in scores.items():
        assert 0.0 <= value <= 1.0, name
