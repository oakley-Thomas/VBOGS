import json

import numpy as np

from scripts.compare_vbgs_vbogs_uncertainty import (
    average_precision,
    auroc,
    evaluate_method,
    run_comparison,
)
from scripts.run_vbgs_vbogs_comparison import (
    TORCH_SERVICE,
    build_steps,
    exec_prefix,
    parse_args,
)
from scripts.split_vbgs_vbogs_points import (
    split_points,
    split_selected_frames,
)
from scripts.train_vbgs_kitti_baseline import aggregate_anchor_scores


def test_split_selected_frames_uses_selected_frame_order():
    train, eval_ = split_selected_frames([10, 20, 30, 40, 50, 60, 70], llffhold=3)

    assert eval_ == [10, 40, 70]
    assert train == [20, 30, 50, 60]


def test_split_points_preserves_npz_columns(tmp_path):
    drive = "drive_sync"
    points_path = tmp_path / "points_world.npz"
    metadata_path = tmp_path / "metadata.json"
    output_root = tmp_path / "comparison"

    np.savez_compressed(
        points_path,
        xyz=np.arange(18, dtype=np.float32).reshape(6, 3),
        rgb=np.arange(18, dtype=np.uint8).reshape(6, 3),
        frame_id=np.array([1, 1, 2, 3, 4, 4], dtype=np.int32),
    )
    metadata_path.write_text(
        json.dumps({"selected_frames": [1, 2, 3, 4]}),
        encoding="utf-8",
    )

    metadata = split_points(
        drive=drive,
        points_world_path=points_path,
        selection_metadata_path=metadata_path,
        output_root=output_root,
        llffhold=2,
    )

    train = np.load(output_root / "points_train.npz")
    eval_ = np.load(output_root / "points_eval.npz")
    assert metadata["train_frames"] == [2, 4]
    assert metadata["eval_frames"] == [1, 3]
    assert train["frame_id"].tolist() == [2, 4, 4]
    assert eval_["frame_id"].tolist() == [1, 1, 3]
    assert train["xyz"].shape == (3, 3)
    assert eval_["rgb"].shape == (3, 3)


def test_aggregate_anchor_scores_uses_packed_assignments():
    result = aggregate_anchor_scores(
        point_scores=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        anchor_offsets=np.array([0, 2, 2, 3], dtype=np.int64),
        point_indices=np.array([0, 2, 1], dtype=np.int64),
    )

    np.testing.assert_allclose(result["raw_mean"], [20.0, np.nan, 20.0])
    np.testing.assert_allclose(result["filled_mean"], [20.0, 20.0, 20.0])
    assert result["assignment_count"].tolist() == [2, 0, 1]


def test_comparison_metrics_rank_high_scores_above_positive_eval_points():
    score = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    eval_count = np.array([0.0, 3.0, 0.0, 1.0], dtype=np.float32)

    row = evaluate_method(
        name="method",
        score=score,
        eval_count=eval_count,
        eval_density=eval_count,
        top_n=2,
    )

    assert auroc(score, eval_count > 0) == 1.0
    assert average_precision(score, eval_count > 0) == 1.0
    assert row["spearman_eval_density"] > 0.8
    assert row["top_n_eval_count_overlap"] == 1.0


def test_run_comparison_writes_report_artifacts(tmp_path):
    root = tmp_path / "comparison"
    train_bucket = root / "m4_train"
    eval_bucket = root / "m4_eval"
    baseline_k = root / "vbgs_global" / "K_2"
    train_bucket.mkdir(parents=True)
    eval_bucket.mkdir(parents=True)
    baseline_k.mkdir(parents=True)

    np.savez_compressed(
        train_bucket / "pts_by_anchor.npz",
        point_counts=np.array([4, 1, 0, 2], dtype=np.int32),
    )
    np.savez_compressed(
        eval_bucket / "pts_by_anchor.npz",
        point_counts=np.array([0, 3, 0, 1], dtype=np.int32),
    )
    np.save(train_bucket / "U.npy", np.array([0.2, 0.9, 0.1, 0.8], dtype=np.float32))
    np.save(baseline_k / "U_baseline.npy", np.array([0.1, 0.7, 0.2, 0.6], dtype=np.float32))

    payload = run_comparison(
        drive="drive_sync",
        comparison_root=root,
        train_bucket_root=train_bucket,
        eval_bucket_root=eval_bucket,
        vbogs_u=train_bucket / "U.npy",
        baseline_root=root / "vbgs_global",
        top_n=2,
        no_plots=True,
    )

    assert (root / "comparison_metrics.json").exists()
    assert (root / "comparison_metrics.csv").exists()
    assert (root / "comparison_summary.md").exists()
    assert {row["method"] for row in payload["methods"]} == {
        "vbogs",
        "count_baseline",
        "global_vbgs_K_2",
    }


def test_comparison_orchestrator_dry_run_builds_torch_and_jax_steps():
    args = parse_args(
        [
            "--drive",
            "drive_sync",
            "--use-service-labels",
            "--label-project",
            "vbogs",
            "--dry-run",
            "--k-sweep",
            "2,3",
            "--no-map-viz",
            "--no-render",
            "--no-plots",
        ]
    )

    steps = build_steps(args)
    names = [step.name for step in steps]
    assert names[:5] == [
        "split-points",
        "bucket-train",
        "bucket-eval",
        "fit-vbogs",
        "compute-vbogs-uncertainty",
    ]
    assert "global-vbgs-K-2" in names
    assert "global-vbgs-K-3" in names
    assert names[-1] == "comparison-report"
    assert steps[0].service == TORCH_SERVICE

    prefix = exec_prefix(args, TORCH_SERVICE)
    assert prefix == [
        "docker",
        "exec",
        "-i",
        "-w",
        "/workspace/VBOGS",
        "<vbogs:vbogs-torch-container-by-label>",
    ]
