import json

import numpy as np
import pytest

from scripts.analyze_experiment04 import (
    analyze,
    ause,
    build_test_name_mapping,
    check_fairness,
    load_variant,
    sparsification_curve,
    spearman,
    wide_only_names,
)

WIDE = "camera_front_wide_120fov"
TELE = "camera_front_tele_30fov"
LLFFHOLD = 8


def image_name(camera_id, frame):
    return f"{camera_id}/{camera_id}_{frame:010d}_{frame:010d}.png"


def stem(camera_id, frame):
    return f"{camera_id}_{frame:010d}_{frame:010d}"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_variant(tmp_path, name, camera_ids, num_frames=16, psnr_by_stem=None):
    """Build a fake bundled variant run dir mirroring the pipeline layout."""
    scene = "pai_test_clip"
    run_dir = tmp_path / name / scene
    frames = list(range(num_frames))

    frame_records = []
    for frame in frames:
        record = {
            "frame_id": frame,
            "primary_camera_id": camera_ids[0],
            "cameras": {
                camera_id: {"image_name": image_name(camera_id, frame)}
                for camera_id in camera_ids
            },
        }
        frame_records.append(record)

    write_json(
        run_dir / "prepared" / "metadata.json",
        {
            "camera_ids": list(camera_ids),
            "primary_camera_id": camera_ids[0],
            "num_frames": num_frames,
            "frame_records": frame_records,
        },
    )
    (run_dir / "octree").mkdir(parents=True, exist_ok=True)
    (run_dir / "octree" / "config.yaml").write_text(
        f"model_params:\n  llffhold: {LLFFHOLD}\n", encoding="utf-8"
    )
    write_json(run_dir / "run_manifest.json", {"source_paths": {"octree_model_path": "/nonexistent"}})

    all_names = sorted(
        image_name(camera_id, frame) for camera_id in camera_ids for frame in frames
    )
    test_stems = [name.rsplit("/", 1)[-1].split(".")[0] for name in all_names[0::LLFFHOLD]]

    psnr_by_stem = psnr_by_stem or {}
    per_view_psnr = {}
    per_view_ssim = {}
    per_view_lpips = {}
    nbv_rows = []
    for index, test_stem in enumerate(test_stems):
        key = f"{index:05d}.png"
        psnr = psnr_by_stem.get(test_stem, 20.0 + index)
        per_view_psnr[key] = psnr
        per_view_ssim[key] = 0.5 + 0.01 * index
        per_view_lpips[key] = 0.5 - 0.01 * index
        # Uncertainty perfectly tracks error: higher for low-PSNR views.
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


def test_build_test_name_mapping_groups_by_camera_and_holds_out_every_eighth():
    names = [
        image_name(camera_id, frame)
        for frame in range(16)
        for camera_id in (WIDE, TELE)
    ]
    test_stems = build_test_name_mapping(names, LLFFHOLD)

    # Sorted order groups tele (t < w) before wide; each 16-frame block
    # starts at an index divisible by 8, so both blocks hold out frames 0, 8.
    assert test_stems == [
        stem(TELE, 0),
        stem(TELE, 8),
        stem(WIDE, 0),
        stem(WIDE, 8),
    ]
    assert wide_only_names(test_stems, WIDE) == [stem(WIDE, 0), stem(WIDE, 8)]


def test_spearman_matches_known_values():
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])) == pytest.approx(1.0)
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([30.0, 20.0, 10.0])) == pytest.approx(-1.0)
    rho = spearman(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 3.0, 2.0, 4.0]))
    assert rho == pytest.approx(0.8)


def test_sparsification_perfect_predictor_matches_oracle():
    errors = np.array([4.0, 1.0, 3.0, 2.0])
    fractions, predicted, oracle = sparsification_curve(errors.copy(), errors)
    np.testing.assert_allclose(predicted, oracle)
    assert ause(predicted, oracle) == pytest.approx(0.0)
    # Removing the most-uncertain (= worst) views lowers the retained mean.
    assert predicted[0] == pytest.approx(errors.mean())
    assert predicted[-1] == pytest.approx(1.0)
    assert fractions[0] == 0.0


def test_sparsification_bad_predictor_has_positive_ause():
    errors = np.array([1.0, 2.0, 3.0, 4.0])
    inverted = -errors
    _, predicted, oracle = sparsification_curve(inverted, errors)
    assert ause(predicted, oracle) > 0.0


def test_load_variant_maps_positional_metrics_to_image_stems(tmp_path):
    make_variant(tmp_path, "cam2", [WIDE, TELE])
    variant = load_variant(tmp_path / "cam2", WIDE)

    assert variant.camera_ids == [WIDE, TELE]
    assert variant.method == "ours_90000"
    assert len(variant.test_names) == 4
    assert variant.test_names[0] == stem(TELE, 0)
    # per_view "00000.png" belongs to the first sorted test view (tele frame 0).
    assert variant.per_view["PSNR"][stem(TELE, 0)] == pytest.approx(20.0)
    assert variant.view_uncertainty[stem(WIDE, 8)] == pytest.approx(100.0 - 23.0)


def test_analyze_two_variants_produces_comparison(tmp_path):
    make_variant(tmp_path, "cam1", [WIDE])
    make_variant(tmp_path, "cam2", [WIDE, TELE])

    comparison = analyze(
        tmp_path,
        variants_filter=None,
        primary_camera=WIDE,
        output_dir=tmp_path / "analysis",
        make_plots=False,
    )

    assert (tmp_path / "analysis" / "comparison.json").exists()
    assert (tmp_path / "analysis" / "metrics.csv").exists()
    assert (tmp_path / "analysis" / "metrics_table.md").exists()

    rows = {row["variant"]: row for row in comparison["variants"]}
    assert rows["cam1"]["cameras"] == 1
    assert rows["cam2"]["cameras"] == 2
    assert rows["cam1"]["wide_test_views"] == 2
    assert rows["cam2"]["wide_test_views"] == 2
    assert comparison["fairness"]["shared_primary_test_views"] == [
        stem(WIDE, 0),
        stem(WIDE, 8),
    ]
    # Uncertainty score is constructed as 100 - PSNR, so calibration is perfect.
    assert comparison["calibration"]["cam1"]["PSNR"]["spearman"] == pytest.approx(1.0)
    assert comparison["calibration"]["cam1"]["PSNR"]["ause"] == pytest.approx(0.0)
    assert comparison["warnings"] == []


def test_analyze_rejects_mismatched_primary_test_frames(tmp_path):
    make_variant(tmp_path, "cam1", [WIDE], num_frames=16)
    make_variant(tmp_path, "cam2", [WIDE, TELE], num_frames=24)

    with pytest.raises(ValueError, match="not comparable"):
        analyze(
            tmp_path,
            variants_filter=None,
            primary_camera=WIDE,
            output_dir=tmp_path / "analysis",
            make_plots=False,
        )


def test_check_fairness_requires_primary_views(tmp_path):
    make_variant(tmp_path, "cam1", [TELE])
    variant = load_variant(tmp_path / "cam1", WIDE)
    with pytest.raises(ValueError, match="no test views from primary camera"):
        check_fairness([variant], WIDE)
