import numpy as np

from scripts.bucket_points import (
    build_level_indices,
    count_point_assignments,
    fill_packed_point_indices,
    normalized_points_from_xyz_rgb,
    normalization_params_from_xyz_rgb,
    select_points,
)


def test_chunked_assignment_counts_and_packs_points():
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
    level_indices = build_level_indices(
        anchor_xyz,
        anchor_level,
        levels=2,
        voxel_size=1.0,
        fork=2,
        init_pos=np.zeros((3,), dtype=np.float32),
    )

    point_counts, level_counts = count_point_assignments(
        points_xyz,
        level_indices,
        np.zeros((3,), dtype=np.float32),
        num_anchors=anchor_xyz.shape[0],
        chunk_size=1,
    )
    anchor_offsets = np.zeros(anchor_xyz.shape[0] + 1, dtype=np.int64)
    anchor_offsets[1:] = np.cumsum(point_counts, dtype=np.int64)
    point_indices = fill_packed_point_indices(
        points_xyz,
        level_indices,
        np.zeros((3,), dtype=np.float32),
        anchor_offsets,
        chunk_size=1,
    )

    assert point_counts.tolist() == [1, 1, 1, 1]
    assert level_counts.tolist() == [2, 2]
    assert point_indices.tolist() == [0, 1, 0, 1]


def test_select_points_uses_even_deterministic_cap():
    xyz = np.arange(30, dtype=np.float32).reshape(10, 3)
    rgb = np.arange(30, dtype=np.uint8).reshape(10, 3)
    frame_id = np.arange(10, dtype=np.int32)

    selected_xyz, selected_rgb, selected_frame_id, metadata = select_points(
        xyz,
        rgb,
        frame_id,
        max_points=4,
    )

    assert selected_xyz[:, 0].tolist() == [0.0, 9.0, 18.0, 27.0]
    assert selected_rgb[:, 0].tolist() == [0, 9, 18, 27]
    assert selected_frame_id.tolist() == [0, 3, 6, 9]
    assert metadata["source_point_count"] == 10
    assert metadata["selected_point_count"] == 4


def test_normalization_helpers_match_expected_columns():
    xyz = np.array([[0.0, 2.0, 4.0], [2.0, 4.0, 6.0]], dtype=np.float32)
    rgb = np.array([[10, 20, 30], [30, 40, 50]], dtype=np.uint8)

    params = normalization_params_from_xyz_rgb(xyz, rgb)
    points_norm = normalized_points_from_xyz_rgb(xyz, rgb, params, chunk_size=1)

    assert params["offset"].tolist() == [1.0, 3.0, 5.0, 20.0, 30.0, 40.0]
    assert params["stdevs"].tolist() == [1.0, 1.0, 1.0, 10.0, 10.0, 10.0]
    np.testing.assert_allclose(
        points_norm,
        np.array(
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
