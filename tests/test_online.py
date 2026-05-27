import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.bucket_points import build_level_indices, count_point_assignments
from scripts.ros2_online_nbv_node import image_msg_to_array, make_stereo_args
from vbogs.online import (
    ONLINE_STATE_VERSION,
    apply_online_exact_fixed_k_update,
    apply_online_moment_update,
    backfill_initial_fields_from_points,
    bucket_points_with_cache,
    build_anchor_grid_cache,
    expand_posterior_to_anchor_rows,
    load_anchor_grid_cache,
    normalize_online_observations,
    rank_candidate_scores,
    save_anchor_grid_cache,
    score_uncertainty_alpha,
)
from vbogs.online.scoring import CandidateScore


def tiny_anchor_fixture():
    anchor_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    anchor_level = np.array([0, 0, 1, 1], dtype=np.int16)
    points_xyz = np.array(
        [
            [0.1, 0.0, 0.0],
            [1.1, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return anchor_xyz, anchor_level, points_xyz


def test_online_bucketing_matches_offline_helpers_on_multilevel_fixture():
    anchor_xyz, anchor_level, points_xyz = tiny_anchor_fixture()
    init_pos = np.zeros((3,), dtype=np.float32)
    level_indices = build_level_indices(
        anchor_xyz,
        anchor_level,
        levels=2,
        voxel_size=1.0,
        fork=2,
        init_pos=init_pos,
    )
    offline_counts, offline_level_counts = count_point_assignments(
        points_xyz,
        level_indices,
        init_pos,
        num_anchors=anchor_xyz.shape[0],
        chunk_size=1,
    )

    cache = build_anchor_grid_cache(
        anchor_xyz=anchor_xyz,
        anchor_level=anchor_level,
        voxel_size=1.0,
        fork=2,
        init_pos=init_pos,
        levels=2,
    )
    online = bucket_points_with_cache(points_xyz, cache, chunk_size=1)

    assert online["point_counts"].tolist() == offline_counts.tolist()
    assert online["level_assignment_counts"].tolist() == offline_level_counts.tolist()
    assert online["touched_anchor_ids"].tolist() == [0, 1, 2, 3]


def test_anchor_grid_cache_round_trips(tmp_path):
    anchor_xyz, anchor_level, points_xyz = tiny_anchor_fixture()
    cache = build_anchor_grid_cache(
        anchor_xyz=anchor_xyz,
        anchor_level=anchor_level,
        voxel_size=1.0,
        fork=2,
        init_pos=np.zeros((3,), dtype=np.float32),
        levels=2,
    )
    path = tmp_path / "anchor_grid_cache.npz"

    save_anchor_grid_cache(path, cache)
    loaded = load_anchor_grid_cache(path)
    bucketed = bucket_points_with_cache(points_xyz, loaded, chunk_size=1)

    assert loaded.anchor_count == 4
    assert bucketed["point_counts"].tolist() == [1, 1, 1, 1]


def test_online_normalization_uses_fixed_params_and_reports_outliers():
    xyz = np.array([[1.0, 2.0, 3.0], [100.0, 2.0, 3.0]], dtype=np.float32)
    rgb = np.array([[10, 20, 30], [10, 200, 30]], dtype=np.uint8)
    norm_params = {
        "offset": np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=np.float32),
        "stdevs": np.ones((6,), dtype=np.float32),
    }

    points_norm, stats = normalize_online_observations(xyz, rgb, norm_params, outlier_z=6.0)

    np.testing.assert_allclose(points_norm[0], np.zeros((6,), dtype=np.float32))
    assert stats["outlier_count"] == 1
    assert stats["point_count"] == 2
    assert stats["max_abs_z"] == 180.0


def make_posterior():
    anchor_count = 3
    observed_anchor_ids = np.array([0, 1], dtype=np.int64)
    alpha = np.ones((2, 2), dtype=np.float32)
    spatial_mean = np.zeros((2, 2, 3, 1), dtype=np.float32)
    spatial_kappa = np.ones((2, 2, 1, 1), dtype=np.float32)
    spatial_u = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 2, 3, 3)).copy()
    spatial_n = np.full((2, 2, 1, 1), 5.0, dtype=np.float32)
    delta_mean = np.zeros((2, 2, 3, 1), dtype=np.float32)
    delta_kappa = np.ones((2, 2, 1, 1), dtype=np.float32)
    delta_u = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 2, 3, 3)).copy()
    delta_n = np.full((2, 2, 1, 1), 5.0, dtype=np.float32)
    return {
        "is_observed": np.array([True, True, False]),
        "observed_anchor_ids": observed_anchor_ids,
        "point_count": np.array([4, 4, 0], dtype=np.int32),
        "final_k": np.array([1, 2], dtype=np.int16),
        "final_elbo": np.array([1.0, 1.0], dtype=np.float32),
        "selected_gain": np.array([0.0, 0.0], dtype=np.float32),
        "under_modeled": np.array([False, False]),
        "fit_completed": np.array([True, True]),
        "fit_batch_size": np.array([4, 4], dtype=np.int32),
        "k_growth_attempted": np.array([False, False]),
        "alpha": alpha,
        "spatial_mean": spatial_mean,
        "spatial_kappa": spatial_kappa,
        "spatial_u": spatial_u,
        "spatial_n": spatial_n,
        "delta_mean": delta_mean,
        "delta_kappa": delta_kappa,
        "delta_u": delta_u,
        "delta_n": delta_n,
        "k_init": np.array(1, dtype=np.int16),
        "k_max": np.array(2, dtype=np.int16),
        "min_points_per_anchor": np.array(2, dtype=np.int32),
    }


def test_online_update_changes_only_touched_observed_anchor():
    state = expand_posterior_to_anchor_rows(make_posterior())
    before_anchor0 = state["alpha"][0].copy()
    before_anchor2 = state["alpha"][2].copy()
    batch = {
        "points_norm": np.array(
            [
                [0.2, 0.1, 0.0, 0.0, 0.0, 0.1],
                [0.3, 0.0, 0.1, 0.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "anchor_offsets": np.array([0, 0, 2, 2], dtype=np.int64),
        "point_indices": np.array([0, 1], dtype=np.int64),
        "touched_anchor_ids": np.array([1], dtype=np.int64),
    }

    updated, metadata = apply_online_moment_update(
        state,
        batch,
        seq=7,
        min_points_per_anchor=2,
    )

    assert metadata["updated_anchor_ids"] == [1]
    assert int(updated["last_update_seq"]) == 7
    np.testing.assert_allclose(updated["alpha"][0], before_anchor0)
    np.testing.assert_allclose(updated["alpha"][2], before_anchor2, equal_nan=True)
    assert not np.allclose(updated["alpha"][1], state["alpha"][1])


def test_online_update_can_initialize_previously_unobserved_anchor():
    state = expand_posterior_to_anchor_rows(make_posterior())
    batch = {
        "points_norm": np.array(
            [
                [1.0, 0.0, 0.0, 10.0, 0.0, 0.0],
                [1.1, 0.0, 0.0, 10.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "anchor_offsets": np.array([0, 0, 0, 2], dtype=np.int64),
        "point_indices": np.array([0, 1], dtype=np.int64),
        "touched_anchor_ids": np.array([2], dtype=np.int64),
    }

    updated, metadata = apply_online_moment_update(
        state,
        batch,
        seq=8,
        min_points_per_anchor=2,
    )

    assert metadata["updated_anchor_ids"] == [2]
    assert bool(updated["is_observed"][2]) is True
    assert bool(updated["fit_completed"][2]) is True
    assert int(updated["final_k"][2]) == 1


def test_online_state_v2_backfills_initial_fields_from_points():
    state = expand_posterior_to_anchor_rows(make_posterior())
    assert int(state["state_version"]) == ONLINE_STATE_VERSION
    assert "initial_spatial_mean" in state
    assert not np.isfinite(state["initial_spatial_mean"][0]).any()

    points_norm = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0, 1.1, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    anchor_offsets = np.array([0, 2, 4, 4], dtype=np.int64)
    point_indices = np.array([0, 1, 2, 3], dtype=np.int64)

    updated = backfill_initial_fields_from_points(
        state,
        points_norm=points_norm,
        anchor_offsets=anchor_offsets,
        point_indices=point_indices,
        seed=0,
    )

    assert np.isfinite(updated["initial_spatial_mean"][0, :1]).all()
    assert np.isfinite(updated["initial_spatial_mean"][1, :2]).all()
    assert not np.isfinite(updated["initial_spatial_mean"][2]).any()


def test_exact_fixed_k_update_changes_only_touched_observed_anchor():
    state = expand_posterior_to_anchor_rows(make_posterior())
    state = backfill_initial_fields_from_points(
        state,
        points_norm=np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.1, 0.0, 0.0, 1.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        anchor_offsets=np.array([0, 2, 4, 4], dtype=np.int64),
        point_indices=np.array([0, 1, 2, 3], dtype=np.int64),
    )
    before_anchor0 = state["alpha"][0].copy()
    before_anchor2 = state["alpha"][2].copy()
    batch = {
        "points_norm": np.array(
            [
                [0.2, 0.1, 0.0, 0.0, 0.0, 0.1],
                [0.3, 0.0, 0.1, 0.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "anchor_offsets": np.array([0, 0, 2, 2], dtype=np.int64),
        "point_indices": np.array([0, 1], dtype=np.int64),
        "touched_anchor_ids": np.array([1], dtype=np.int64),
    }

    updated, metadata = apply_online_exact_fixed_k_update(
        state,
        batch,
        seq=9,
        min_points_per_anchor=2,
    )

    assert metadata["update_mode"] == "exact_fixed_k"
    assert metadata["updated_anchor_ids"] == [1]
    assert int(updated["last_update_seq"]) == 9
    np.testing.assert_allclose(updated["alpha"][0], before_anchor0)
    np.testing.assert_allclose(updated["alpha"][2], before_anchor2, equal_nan=True)
    assert not np.allclose(updated["alpha"][1], state["alpha"][1])


def test_exact_fixed_k_update_defers_then_initializes_unobserved_anchor():
    state = expand_posterior_to_anchor_rows(make_posterior())
    batch_one = {
        "points_norm": np.array([[1.0, 0.0, 0.0, 10.0, 0.0, 0.0]], dtype=np.float32),
        "anchor_offsets": np.array([0, 0, 0, 1], dtype=np.int64),
        "point_indices": np.array([0], dtype=np.int64),
        "touched_anchor_ids": np.array([2], dtype=np.int64),
    }
    deferred, metadata = apply_online_exact_fixed_k_update(
        state,
        batch_one,
        seq=10,
        min_points_per_anchor=2,
    )

    assert metadata["deferred_anchor_ids"] == [2]
    assert bool(deferred["fit_completed"][2]) is False
    assert deferred["pending_points_norm"].shape[0] == 1

    batch_two = {
        "points_norm": np.array([[1.1, 0.0, 0.0, 10.1, 0.0, 0.0]], dtype=np.float32),
        "anchor_offsets": np.array([0, 0, 0, 1], dtype=np.int64),
        "point_indices": np.array([0], dtype=np.int64),
        "touched_anchor_ids": np.array([2], dtype=np.int64),
    }
    initialized, metadata = apply_online_exact_fixed_k_update(
        deferred,
        batch_two,
        seq=11,
        min_points_per_anchor=2,
    )

    assert metadata["updated_anchor_ids"] == [2]
    assert bool(initialized["fit_completed"][2]) is True
    assert bool(initialized["is_observed"][2]) is True
    assert initialized["pending_points_norm"].shape[0] == 0


def test_score_uncertainty_alpha_and_ranking():
    score, unc_sum, alpha_sum = score_uncertainty_alpha(
        np.array([[2.0, 4.0]], dtype=np.float32),
        np.array([[1.0, 2.0]], dtype=np.float32),
        eps=1.0,
    )

    assert unc_sum == 6.0
    assert alpha_sum == 3.0
    assert score == 1.5
    ranked = rank_candidate_scores(
        [
            CandidateScore(candidate_index=0, score=0.1, unc_sum=1.0, alpha_sum=10.0),
            CandidateScore(candidate_index=1, score=0.5, unc_sum=1.0, alpha_sum=2.0),
        ]
    )
    assert [row.candidate_index for row in ranked] == [1, 0]


def test_ros_image_conversion_and_stereo_config_helpers_do_not_require_ros():
    msg = SimpleNamespace(
        height=1,
        width=2,
        encoding="rgb8",
        data=np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8).tobytes(),
    )
    arr = image_msg_to_array(msg)

    assert arr.shape == (1, 2, 3)
    assert arr[0, 1].tolist() == [4, 5, 6]
    args = make_stereo_args({"stereo": {"matcher": "sgbm", "pixel_step": 4}})
    assert args.matcher == "sgbm"
    assert args.pixel_step == 4


def test_ros2_runtime_dependency_is_explicit_when_unavailable():
    if importlib.util.find_spec("rclpy") is not None:
        pytest.skip("rclpy is installed; missing-runtime path is not active")

    assert importlib.util.find_spec("rclpy") is None
