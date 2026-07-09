import json

import numpy as np
import pytest

from scripts.analyze_experiment05 import (
    analyze,
    check_fairness,
    image_names_from_prepared_metadata,
    load_variant,
    seed_delta,
)

LLFFHOLD = 8
WIDE = "camera_front_wide_120fov"
TELE = "camera_front_tele_30fov"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def kitti_image_name(frame):
    return f"{frame:010d}.png"


def ncore_image_name(camera_id, frame):
    return f"{camera_id}/{camera_id}_{frame:010d}_{frame:010d}.png"


def kitti_frame_records(frames):
    return [
        {
            "frame_id": frame,
            "images": [
                {
                    "camera": "image_00",
                    "camera_id": 1,
                    "image_name": kitti_image_name(frame),
                }
            ],
        }
        for frame in frames
    ]


def ncore_frame_records(frames, camera_ids):
    return [
        {
            "frame_id": frame,
            "primary_camera_id": camera_ids[0],
            "cameras": {
                camera_id: {"image_name": ncore_image_name(camera_id, frame)}
                for camera_id in camera_ids
            },
        }
        for frame in frames
    ]


def make_variant(
    tmp_path,
    name,
    *,
    seed_mode,
    seed_points,
    shape="kitti",
    num_frames=16,
    psnr_offset=0.0,
):
    """Build a fake bundled variant run dir mirroring the pipeline layout."""
    scene = "test_scene"
    run_dir = tmp_path / name / scene
    frames = list(range(num_frames))

    if shape == "kitti":
        frame_records = kitti_frame_records(frames)
        all_names = sorted(kitti_image_name(frame) for frame in frames)
    else:
        frame_records = ncore_frame_records(frames, [WIDE, TELE])
        all_names = sorted(
            ncore_image_name(camera_id, frame)
            for camera_id in (WIDE, TELE)
            for frame in frames
        )

    write_json(
        run_dir / "prepared" / "metadata.json",
        {
            "num_frames": num_frames,
            "seed_mode": seed_mode,
            "seed_metadata": {
                "seed_source": "velodyne" if seed_mode == "lidar" else "stereo_sgbm",
                "seed_point_count": seed_points,
            },
            "frame_records": frame_records,
        },
    )
    (run_dir / "octree").mkdir(parents=True, exist_ok=True)
    (run_dir / "octree" / "config.yaml").write_text(
        f"model_params:\n  llffhold: {LLFFHOLD}\n", encoding="utf-8"
    )
    write_json(
        run_dir / "run_manifest.json",
        {"source_paths": {"octree_model_path": "/nonexistent"}},
    )

    test_stems = [
        name.rsplit("/", 1)[-1].split(".")[0] for name in all_names[0::LLFFHOLD]
    ]

    per_view_psnr = {}
    per_view_ssim = {}
    per_view_lpips = {}
    nbv_rows = []
    for index, test_stem in enumerate(test_stems):
        key = f"{index:05d}.png"
        psnr = 20.0 + index + psnr_offset
        per_view_psnr[key] = psnr
        per_view_ssim[key] = 0.5 + 0.01 * index
        per_view_lpips[key] = 0.5 - 0.01 * index
        nbv_rows.append(
            {
                "rank": index + 1,
                "candidate_index": index,
                "image_name": test_stem,
                "score": 100.0 - psnr,
                "unc_sum": 100.0 - psnr,
                "alpha_sum": 1.0,
            }
        )

    method = "ours_90000"
    write_json(
        run_dir / "octree" / "results.json",
        {
            method: {
                "PSNR": float(np.mean(list(per_view_psnr.values()))),
                "SSIM": float(np.mean(list(per_view_ssim.values()))),
                "LPIPS": float(np.mean(list(per_view_lpips.values()))),
                "GS_NUMS": 1000.0,
            }
        },
    )
    write_json(
        run_dir / "octree" / "per_view.json",
        {
            method: {
                "PSNR": per_view_psnr,
                "SSIM": per_view_ssim,
                "LPIPS": per_view_lpips,
            }
        },
    )
    write_json(
        run_dir / "nbv" / "nbv_scores.json",
        {"candidate_count": len(nbv_rows), "top_k": nbv_rows},
    )

    uncertainty = np.linspace(1.0, 2.0, 32).astype(np.float32)
    (run_dir / "uncertainty").mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "uncertainty" / "U.npy", uncertainty)
    write_json(
        run_dir / "uncertainty" / "uncertainty_metadata.json",
        {
            "anchor_count": 32,
            "observed_anchor_count": 24,
            "observed_summary": {"mean": 1.4},
        },
    )
    return run_dir


def test_image_names_support_both_prepared_metadata_shapes():
    kitti_metadata = {"frame_records": kitti_frame_records([0, 1])}
    assert image_names_from_prepared_metadata(kitti_metadata) == [
        "0000000000.png",
        "0000000001.png",
    ]

    ncore_metadata = {"frame_records": ncore_frame_records([0], [WIDE, TELE])}
    assert sorted(image_names_from_prepared_metadata(ncore_metadata)) == [
        ncore_image_name(TELE, 0),
        ncore_image_name(WIDE, 0),
    ]


def test_load_variant_reads_seed_fields_from_kitti_bundle(tmp_path):
    make_variant(tmp_path, "lidar", seed_mode="lidar", seed_points=60000)
    variant = load_variant(tmp_path / "lidar")

    assert variant.seed_mode == "lidar"
    assert variant.seed_points == 60000
    assert variant.method == "ours_90000"
    assert len(variant.test_names) == 2
    assert variant.per_view["PSNR"][variant.test_names[0]] == pytest.approx(20.0)


def test_load_variant_reads_ncore_shaped_bundle(tmp_path):
    make_variant(
        tmp_path, "sgbm", seed_mode="stereo", seed_points=40000, shape="ncore"
    )
    variant = load_variant(tmp_path / "sgbm")

    assert variant.seed_mode == "stereo"
    assert variant.seed_points == 40000
    assert len(variant.test_names) == 4


def test_analyze_two_seed_variants_produces_comparison_with_delta(tmp_path):
    make_variant(tmp_path, "sgbm", seed_mode="stereo", seed_points=40000)
    make_variant(
        tmp_path, "lidar", seed_mode="lidar", seed_points=60000, psnr_offset=2.0
    )

    comparison = analyze(
        tmp_path,
        variants_filter=None,
        output_dir=tmp_path / "analysis",
        make_plots=False,
    )

    assert (tmp_path / "analysis" / "comparison.json").exists()
    assert (tmp_path / "analysis" / "metrics.csv").exists()
    assert (tmp_path / "analysis" / "metrics_table.md").exists()

    rows = {row["variant"]: row for row in comparison["variants"]}
    assert rows["sgbm"]["seed_mode"] == "stereo"
    assert rows["sgbm"]["seed_points"] == 40000
    assert rows["lidar"]["seed_mode"] == "lidar"
    assert rows["lidar"]["seed_points"] == 60000
    assert comparison["fairness"]["shared_test_view_count"] == 2
    assert comparison["delta_lidar_minus_sgbm"]["PSNR"] == pytest.approx(2.0)
    # Uncertainty score is constructed as 100 - PSNR, so calibration is perfect.
    assert comparison["calibration"]["sgbm"]["PSNR"]["spearman"] == pytest.approx(1.0)
    assert comparison["warnings"] == []


def test_analyze_rejects_mismatched_test_frames(tmp_path):
    make_variant(tmp_path, "sgbm", seed_mode="stereo", seed_points=40000, num_frames=16)
    make_variant(tmp_path, "lidar", seed_mode="lidar", seed_points=60000, num_frames=24)

    with pytest.raises(ValueError, match="not comparable"):
        analyze(
            tmp_path,
            variants_filter=None,
            output_dir=tmp_path / "analysis",
            make_plots=False,
        )


def test_check_fairness_rejects_empty_test_views(tmp_path):
    make_variant(tmp_path, "sgbm", seed_mode="stereo", seed_points=40000)
    variant = load_variant(tmp_path / "sgbm")
    variant.test_names = []
    with pytest.raises(ValueError, match="no held-out test views"):
        check_fairness([variant])


def test_seed_delta_requires_both_arms():
    assert seed_delta([{"variant": "sgbm", "PSNR": 20.0}]) is None
