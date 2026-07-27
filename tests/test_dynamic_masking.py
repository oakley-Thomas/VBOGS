import numpy as np
import pytest

from vbogs.dynamic_masking import (
    InstanceObservation,
    MotionTrack,
    associate_world_tracks,
    dilate_dynamic_mask,
    filter_moving_cuboid_points,
    is_confirmed_moving,
    read_static_mask,
    write_manifest,
    write_static_mask,
)


def test_paired_analysis_rejects_mismatched_view_sets_and_reports_metric_delta():
    from scripts.analyze_dynamic_masking_experiment import paired_rows

    base = {
        "view_id": "wide/000.png", "frame_id": 0, "camera_id": "wide", "is_primary": True,
        "static_pixel_fraction": 0.9, "PSNR": 20.0, "SSIM": 0.8, "LPIPS": 0.2,
        "PSNR_static": 21.0, "SSIM_static": 0.81, "LPIPS_static": 0.19,
        "uncertainty_score_static": 1.0,
    }
    masked = {**base, "PSNR_static": 22.0, "index": 0}
    unmasked = {**base, "index": 0}
    rows = paired_rows({"per_view": {"views": [unmasked]}}, {"per_view": {"views": [masked]}})
    assert rows[0]["delta_PSNR_static"] == 1.0
    with pytest.raises(ValueError, match="view IDs"):
        paired_rows({"per_view": {"views": [unmasked]}}, {"per_view": {"views": [{**masked, "view_id": "other.png"}]}})


def observation(frame, time, position):
    return InstanceObservation(str(frame), time, 3, np.ones((2, 2), bool), np.asarray(position), 1.0)


def test_mask_artifact_polarity_and_shape_validation(tmp_path):
    write_manifest(tmp_path, {"dataset": "kitti360"})
    write_static_mask(tmp_path, "image_00/000.png", np.array([[True, False]], dtype=bool))

    assert read_static_mask(tmp_path, "image_00/000.png", (1, 2)).tolist() == [[True, False]]
    try:
        read_static_mask(tmp_path, "image_00/000.png", (2, 2))
    except ValueError as exc:
        assert "expected" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("shape mismatch should fail")


def test_dilation_and_confirmed_world_motion():
    mask = np.zeros((5, 5), bool)
    mask[2, 2] = True
    assert dilate_dynamic_mask(mask, 1).sum() > 1

    track = MotionTrack(1, 3, [
        observation(0, 0.0, (0, 0, 0)),
        observation(1, 0.5, (0.6, 0, 0)),
        observation(2, 1.0, (1.2, 0, 0)),
    ])
    assert is_confirmed_moving(track)
    tracks = associate_world_tracks(track.observations)
    assert len(tracks) == 1


def test_unreliable_or_short_track_is_retained():
    track = MotionTrack(1, 3, [observation(0, 0.0, (0, 0, 0)), observation(1, 0.1, (3, 0, 0))])
    assert not is_confirmed_moving(track)


def test_ncore_cuboid_filters_only_time_aligned_points(tmp_path):
    write_manifest(tmp_path, {
        "dataset": "nvidia_ncore",
        "moving_cuboids": [{
            "track_id": "actor", "timestamp_us": 1000,
            "center_world": [0.0, 0.0, 0.0], "size_m": [2.0, 2.0, 2.0],
        }],
    })
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    assert filter_moving_cuboid_points(points, tmp_path, 1000).tolist() == [False, True]
    assert filter_moving_cuboid_points(points, tmp_path, 999999).tolist() == [True, True]
