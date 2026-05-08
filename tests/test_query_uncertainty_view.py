import numpy as np
import pytest

from scripts.query_uncertainty_view import (
    matrix_from_values,
    pose_to_c2w,
    project_points_to_camera,
    rendered_anchor_mask_from_gaussians,
    safe_output_stem,
)


def test_matrix_from_values_accepts_quoted_row_major_pose():
    matrix = matrix_from_values(["1,0,0,4 0,1,0,5 0,0,1,6 0,0,0,1"])

    assert matrix.shape == (4, 4)
    assert np.allclose(matrix[:3, 3], [4.0, 5.0, 6.0])


def test_safe_output_stem_removes_path_separators_and_spaces():
    assert safe_output_stem("nested/my query pose.png") == "my_query_pose"


def test_pose_to_c2w_inverts_w2c_matrix():
    w2c = np.eye(4)
    w2c[:3, 3] = [0.0, 0.0, -2.0]

    c2w = pose_to_c2w(w2c, "w2c")

    assert np.allclose(c2w[:3, 3], [0.0, 0.0, 2.0])


def test_project_points_to_camera_marks_geometric_view_cone():
    points_world = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 0.0, -1.0],
            [5.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )

    projected = project_points_to_camera(
        points_world,
        np.eye(4),
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        width=100,
        height=100,
        near=0.1,
        far=10.0,
    )

    assert projected.in_frustum.tolist() == [True, False, False, False]
    assert projected.pixel_xy[0].tolist() == [50.0, 50.0]
    assert projected.camera_xyz[0, 2] == pytest.approx(2.0)


def test_rendered_anchor_mask_from_gaussians_maps_back_to_parent_anchors():
    visible_mask = np.array([False, True, True, False])
    selection_mask = np.array([True, False, True, True, False, True])
    visibility_filter = np.array([False, True, True, False])

    rendered = rendered_anchor_mask_from_gaussians(
        visible_mask,
        selection_mask,
        visibility_filter,
        n_offsets=3,
    )

    assert rendered.tolist() == [False, True, True, False]
