import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vbogs.uncertainty_evaluation import (
    calibration_summary,
    directory_sha256,
    expected_training_image_count,
    evaluation_summary,
    evaluation_views,
    file_sha256,
    normalized_errors,
    psnr_unit_range,
    select_octree_candidate,
    select_uncertainty_candidate,
    split_fingerprint,
    validate_split_integrity,
    verify_selection_lock,
)
from scripts.analyze_uncertainty_evaluation import report, select_octree, select_uncertainty


def kitti_metadata():
    records = []
    splits = {"train": [0], "validation": [1], "test": [2]}
    for frame_id, split in ((0, "train"), (1, "validation"), (2, "test")):
        records.append(
            {
                "frame_id": frame_id,
                "split": split,
                "images": [
                    {
                        "camera": "image_00",
                        "image_name": f"image_00/{frame_id:010d}.png",
                    },
                    {
                        "camera": "image_01",
                        "image_name": f"image_01/{frame_id:010d}.png",
                    },
                ],
            }
        )
    return {"dataset": "kitti360", "frame_splits": splits, "frame_records": records}


def ncore_metadata():
    camera_ids = ["wide", "tele"]
    records = []
    splits = {"train": [10], "validation": [20], "test": [30]}
    for frame_id, split in ((10, "train"), (20, "validation"), (30, "test")):
        records.append(
            {
                "frame_id": frame_id,
                "split": split,
                "cameras": {
                    camera: {"image_name": f"{camera}/{camera}_{frame_id}.png"}
                    for camera in camera_ids
                },
            }
        )
    return {
        "dataset": "nvidia_ncore",
        "primary_camera_id": "wide",
        "camera_ids": camera_ids,
        "frame_splits": splits,
        "frame_records": records,
    }


def metric_rows():
    return [
        {"is_primary": True, "PSNR": 30.0, "SSIM": 0.9, "LPIPS": 0.1, "uncertainty_score": 1.0},
        {"is_primary": True, "PSNR": 20.0, "SSIM": 0.7, "LPIPS": 0.3, "uncertainty_score": 3.0},
        {"is_primary": False, "PSNR": 25.0, "SSIM": 0.8, "LPIPS": 0.2, "uncertainty_score": 2.0},
    ]


def test_split_integrity_and_kitti_primary_mapping():
    metadata = kitti_metadata()
    assert validate_split_integrity(metadata)["validation"] == [1]
    views = evaluation_views(metadata, "validation")
    assert [view.camera_id for view in views] == ["image_00", "image_01"]
    assert [view.is_primary for view in views] == [True, False]
    assert split_fingerprint(metadata) == split_fingerprint(json.loads(json.dumps(metadata)))
    assert expected_training_image_count(metadata) == 2


def test_ncore_mapping_uses_declared_camera_order_and_primary():
    metadata = ncore_metadata()
    views = evaluation_views(metadata, "test")
    assert [view.camera_id for view in views] == ["wide", "tele"]
    assert views[0].is_primary and not views[1].is_primary
    assert expected_training_image_count(metadata) == 2


def test_split_integrity_rejects_leakage():
    metadata = kitti_metadata()
    metadata["frame_splits"]["test"].append(0)
    with pytest.raises(ValueError, match="both train and test"):
        validate_split_integrity(metadata)


def test_metric_summary_and_calibration_rank_worst_view_first():
    rows = metric_rows()
    summary = evaluation_summary(rows)
    assert summary["groups"]["primary"]["metrics"]["PSNR"]["mean"] == pytest.approx(25.0)
    calibration = summary["groups"]["primary"]["calibration"]
    assert calibration["mean_normalized_AUSE"] == pytest.approx(0.0)
    assert calibration["mean_spearman"] == pytest.approx(1.0)


def test_normalized_errors_omits_constant_metric():
    rows = [
        {"PSNR": 10.0, "SSIM": 0.5, "LPIPS": 0.2, "uncertainty_score": 1.0},
        {"PSNR": 10.0, "SSIM": 0.6, "LPIPS": 0.3, "uncertainty_score": 2.0},
    ]
    errors, warning = normalized_errors(rows, "PSNR")
    assert errors is None
    assert "constant" in warning
    assert "PSNR" not in calibration_summary(rows)["metrics"]


def test_octree_selection_uses_declared_lexicographic_order():
    def candidate(name, order, iteration, psnr, ssim, lpips):
        return {
            "profile": name,
            "profile_order": order,
            "iteration": iteration,
            "summary": {
                "groups": {
                    "primary": {
                        "metrics": {
                            "PSNR": {"mean": psnr},
                            "SSIM": {"mean": ssim},
                            "LPIPS": {"mean": lpips},
                        }
                    }
                }
            },
        }

    selected = select_octree_candidate(
        [
            candidate("late", 0, 90000, 28.0, 0.9, 0.1),
            candidate("early", 1, 60000, 28.0, 0.9, 0.1),
            candidate("worse", 2, 30000, 27.9, 0.95, 0.05),
        ]
    )
    assert selected["profile"] == "early"


def test_uncertainty_selection_uses_ause_then_spearman_then_order():
    def candidate(name, order, ause_value, rho):
        return {
            "profile": name,
            "profile_order": order,
            "summary": {
                "groups": {
                    "primary": {
                        "calibration": {
                            "mean_normalized_AUSE": ause_value,
                            "mean_spearman": rho,
                        }
                    }
                }
            },
        }

    selected = select_uncertainty_candidate(
        [candidate("a", 0, 0.2, 0.9), candidate("b", 1, 0.1, 0.5), candidate("c", 2, 0.1, 0.4)]
    )
    assert selected["profile"] == "b"


def test_selection_lock_hashes_files_and_directories(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("value: 1\n", encoding="utf-8")
    model = tmp_path / "iteration_1"
    model.mkdir()
    (model / "point_cloud.ply").write_bytes(b"model")
    lock = {
        "config_hash": "config-hash",
        "split_hash": "split-hash",
        "artifact_hashes": {
            "config": file_sha256(config),
            "model": directory_sha256(model),
        },
    }
    verify_selection_lock(
        lock,
        config_hash="config-hash",
        split_hash="split-hash",
        artifact_paths={"config": config, "model": model},
    )
    (model / "point_cloud.ply").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model"):
        verify_selection_lock(
            lock,
            config_hash="config-hash",
            split_hash="split-hash",
            artifact_paths={"config": config, "model": model},
        )


def test_analyzer_selects_validation_candidates_and_writes_immutable_lock(tmp_path):
    split_hash = "split-hash"
    config_hash = "config-hash"
    model = tmp_path / "model"
    iteration_dir = model / "point_cloud" / "iteration_10000"
    iteration_dir.mkdir(parents=True)
    (model / "config.yaml").write_text("model: true\n", encoding="utf-8")
    (iteration_dir / "point_cloud_anchor.ply").write_bytes(b"model")
    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "kitti360",
                "scene_id": "scene",
                "config_hash": config_hash,
                "split_hash": split_hash,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "octree_runs.json").write_text(
        json.dumps(
            {
                "profiles": [
                    {"name": "profile", "model_path": str(model), "iterations": [10000]}
                ]
            }
        ),
        encoding="utf-8",
    )
    octree_eval = tmp_path / "validation" / "octree" / "profile" / "iteration_10000"
    octree_eval.mkdir(parents=True)
    quality_summary = evaluation_summary(metric_rows())
    (octree_eval / "summary.json").write_text(
        json.dumps({"split": "validation", "split_hash": split_hash, **quality_summary}),
        encoding="utf-8",
    )
    select_octree(tmp_path)

    profiles = []
    for name, scores in (("good", [1.0, 3.0]), ("bad", [3.0, 1.0])):
        profile_root = tmp_path / "work" / name
        profile_root.mkdir(parents=True)
        posterior = profile_root / "anchor_posterior.npz"
        uncertainty = profile_root / "U.npy"
        posterior.write_bytes(name.encode())
        np.save(uncertainty, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
        profiles.append(
            {"name": name, "posterior_path": str(posterior), "u_path": str(uncertainty)}
        )
        rows = [
            {
                "view_id": "v0",
                "is_primary": True,
                "PSNR": 30.0,
                "SSIM": 0.9,
                "LPIPS": 0.1,
                "uncertainty_score": scores[0],
            },
            {
                "view_id": "v1",
                "is_primary": True,
                "PSNR": 20.0,
                "SSIM": 0.7,
                "LPIPS": 0.3,
                "uncertainty_score": scores[1],
            },
        ]
        evaluation = tmp_path / "validation" / "uncertainty" / name
        evaluation.mkdir(parents=True)
        (evaluation / "per_view.json").write_text(json.dumps({"views": rows}), encoding="utf-8")
        (evaluation / "summary.json").write_text(
            json.dumps(
                {
                    "split": "validation",
                    "split_hash": split_hash,
                    **evaluation_summary(rows),
                }
            ),
            encoding="utf-8",
        )
    (tmp_path / "uncertainty_runs.json").write_text(
        json.dumps({"profiles": profiles}), encoding="utf-8"
    )
    select_uncertainty(tmp_path)
    lock = json.loads((tmp_path / "selection.lock.json").read_text(encoding="utf-8"))
    assert lock["uncertainty_profile"] == "good"
    with pytest.raises(FileExistsError, match="immutable"):
        select_uncertainty(tmp_path)

    test_rows = [
        {
            "view_id": "t0",
            "is_primary": True,
            "PSNR": 30.0,
            "SSIM": 0.9,
            "LPIPS": 0.1,
            "uncertainty_score": 1.0,
        },
        {
            "view_id": "t1",
            "is_primary": True,
            "PSNR": 20.0,
            "SSIM": 0.7,
            "LPIPS": 0.3,
            "uncertainty_score": 3.0,
        },
    ]
    test_root = tmp_path / "test"
    test_root.mkdir()
    (test_root / "per_view.json").write_text(json.dumps({"views": test_rows}), encoding="utf-8")
    (test_root / "summary.json").write_text(
        json.dumps(evaluation_summary(test_rows)), encoding="utf-8"
    )
    report(tmp_path)
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "plots" / "test_calibration_scatter.png").exists()


def test_reference_psnr_matches_known_unit_range_value():
    ground_truth = np.zeros((3, 2, 2), dtype=np.float32)
    render = np.full_like(ground_truth, 0.1)
    assert psnr_unit_range(render, ground_truth) == pytest.approx(20.0)


def test_reference_psnr_matches_upstream_octree_when_torch_is_available():
    torch = pytest.importorskip("torch")
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "Octree-AnyGS"))
    from utils.image_utils import psnr

    ground_truth = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    render = torch.full_like(ground_truth, 0.1)
    assert float(psnr(render, ground_truth).mean()) == pytest.approx(
        psnr_unit_range(render.numpy(), ground_truth.numpy()), abs=1.0e-5
    )


def test_gpu_metric_adapter_accepts_channel_permuted_tensors_when_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "Octree-AnyGS"))
    from scripts.evaluate_uncertainty_views import metric_values

    class ZeroLpips:
        def __call__(self, render, ground_truth):
            return torch.zeros((render.shape[0],), dtype=render.dtype, device=render.device)

    render = torch.zeros((2, 2, 3), dtype=torch.float32).permute(2, 0, 1)
    ground_truth = torch.full((2, 2, 3), 0.1, dtype=torch.float32).permute(2, 0, 1)
    assert not render.is_contiguous()
    values = metric_values(render, ground_truth, ZeroLpips())
    assert values["PSNR"] == pytest.approx(20.0, abs=1.0e-5)


@pytest.mark.parametrize("dataset,scene", [("kitti360", "drive"), ("nvidia_ncore", "clip")])
def test_runner_smoke_dry_run_covers_both_datasets(dataset, scene):
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/uncertainty-evaluation",
            "--dataset-name",
            dataset,
            "--scene-id",
            scene,
            "--run-id",
            "pytest-dry-run",
            "--smoke",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "--frame-split train" in completed.stdout
    assert "--split validation" in completed.stdout
    assert "--split test" in completed.stdout
    assert "--no-eval" in completed.stdout


def test_export_uncertainty_ply_colors_anchors(tmp_path):
    plyfile = pytest.importorskip("plyfile")
    pytest.importorskip("matplotlib")

    anchor_ply = tmp_path / "point_cloud_anchor.ply"
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    elements = np.empty(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")])
    elements["x"], elements["y"], elements["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements["opacity"] = 0.0
    plyfile.PlyData([plyfile.PlyElement.describe(elements, "vertex")]).write(anchor_ply)

    u_path = tmp_path / "U.npy"
    np.save(u_path, np.array([0.0, 0.5, 1.0], dtype=np.float32))
    output = tmp_path / "uncertainty_anchors.ply"

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "export_uncertainty_ply.py"),
            "--anchor-ply", str(anchor_ply),
            "--u-path", str(u_path),
            "--output", str(output),
        ],
        check=True,
    )

    written = plyfile.PlyData.read(output)["vertex"]
    assert np.allclose(np.asarray(written["x"]), xyz[:, 0])
    assert np.allclose(np.asarray(written["uncertainty"]), [0.0, 0.5, 1.0])
    colors = np.stack([np.asarray(written[c]) for c in ("red", "green", "blue")], axis=1)
    assert colors.dtype == np.uint8
    assert not np.array_equal(colors[0], colors[2])


def test_export_uncertainty_ply_rejects_length_mismatch(tmp_path):
    plyfile = pytest.importorskip("plyfile")

    anchor_ply = tmp_path / "point_cloud_anchor.ply"
    elements = np.zeros(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    plyfile.PlyData([plyfile.PlyElement.describe(elements, "vertex")]).write(anchor_ply)
    u_path = tmp_path / "U.npy"
    np.save(u_path, np.zeros(2, dtype=np.float32))

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "export_uncertainty_ply.py"),
            "--anchor-ply", str(anchor_ply),
            "--u-path", str(u_path),
            "--output", str(tmp_path / "out.ply"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "does not match the anchor count" in result.stderr
