import importlib.machinery
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
    view_score_fields,
)
from scripts.analyze_uncertainty_evaluation import report, select_octree, select_uncertainty


def load_runner_module():
    """Import `scripts/uncertainty-evaluation`, which has no `.py` suffix."""

    path = Path(__file__).resolve().parents[1] / "scripts" / "uncertainty-evaluation"
    loader = importlib.machinery.SourceFileLoader("uncertainty_evaluation_runner", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


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


def test_calibration_summary_honors_score_key_and_evaluation_adds_observed_baselines():
    rows = [
        {
            "is_primary": True, "frame_id": index, "PSNR": psnr, "SSIM": ssim, "LPIPS": lpips,
            "uncertainty_score": legacy, "uncertainty_score_observed": observed,
            "unobserved_fraction": 0.25,
        }
        for index, (psnr, ssim, lpips, legacy, observed) in enumerate(
            [(30.0, 0.9, 0.1, 3.0, 1.0), (20.0, 0.7, 0.3, 1.0, 3.0)]
        )
    ]
    assert calibration_summary(rows, score_key="uncertainty_score_observed")["mean_spearman"] == pytest.approx(1.0)
    summary = evaluation_summary(rows)["groups"]["primary"]
    assert "calibration_observed" in summary
    assert "frame_id_null" in summary
    assert summary["unobserved_fraction"]["mean"] == pytest.approx(0.25)
    rows[1]["uncertainty_score_observed"] = None
    assert "calibration_observed" not in evaluation_summary(rows)["groups"]["primary"]


def test_view_score_fields_handles_zero_mass_and_clamps_coverage():
    fields = view_score_fields(10.0, 4.0, 2.0, 5.0)
    assert fields["uncertainty_score_observed"] == pytest.approx(2.0)
    assert fields["unobserved_fraction"] == pytest.approx(0.6)
    zero = view_score_fields(0.0, 0.0, 0.0, 0.0)
    assert zero["uncertainty_score_observed"] == 0.0
    assert zero["unobserved_fraction"] == 1.0


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


def test_uncertainty_selection_prefers_observed_calibration_and_falls_back_to_legacy():
    def candidate(name, order, legacy, observed=None):
        primary = {"calibration": {"mean_normalized_AUSE": legacy, "mean_spearman": 0.5}}
        if observed is not None:
            primary["calibration_observed"] = {
                "mean_normalized_AUSE": observed,
                "mean_spearman": 0.5,
            }
        return {"profile": name, "profile_order": order, "summary": {"groups": {"primary": primary}}}

    selected = select_uncertainty_candidate(
        [candidate("legacy", 0, 0.01), candidate("observed", 1, 0.9, observed=0.005)]
    )
    assert selected["profile"] == "observed"


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
                "uncertainty_score_observed": scores[0],
                "unobserved_fraction": 0.25,
                "frame_id": 0,
            },
            {
                "view_id": "v1",
                "is_primary": True,
                "PSNR": 20.0,
                "SSIM": 0.7,
                "LPIPS": 0.3,
                "uncertainty_score": scores[1],
                "uncertainty_score_observed": scores[1],
                "unobserved_fraction": 0.25,
                "frame_id": 1,
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
    metrics = (tmp_path / "validation" / "uncertainty_metrics.md").read_text(encoding="utf-8")
    assert "Observed mean normalized AUSE" in metrics
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
            "uncertainty_score_observed": 1.0,
            "unobserved_fraction": 0.25,
            "frame_id": 0,
        },
        {
            "view_id": "t1",
            "is_primary": True,
            "PSNR": 20.0,
            "SSIM": 0.7,
            "LPIPS": 0.3,
            "uncertainty_score": 3.0,
            "uncertainty_score_observed": 3.0,
            "unobserved_fraction": 0.25,
            "frame_id": 1,
        },
    ]
    test_root = tmp_path / "test"
    test_root.mkdir()
    (test_root / "per_view.json").write_text(json.dumps({"views": test_rows}), encoding="utf-8")
    (test_root / "summary.json").write_text(
        json.dumps(evaluation_summary(test_rows)), encoding="utf-8"
    )
    pytest.importorskip("matplotlib")
    report(tmp_path)
    assert (tmp_path / "report.md").exists()
    report_text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Observed AUSE" in report_text
    assert "Frame-id null rho" in report_text
    assert "Mean unobserved fraction" in report_text
    assert (tmp_path / "plots" / "test_calibration_scatter.png").exists()
    assert (tmp_path / "plots" / "test_calibration_scatter_legacy.png").exists()


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


def test_static_image_metrics_and_mask_loader_when_torch_is_available(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image
    from vbogs.dynamic_masking import write_static_mask
    from scripts.evaluate_uncertainty_views import load_static_mask_for_evaluation, static_metric_values

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "Octree-AnyGS"))

    class ZeroLpips:
        def __call__(self, render, ground_truth):
            return torch.zeros((render.shape[0],), dtype=render.dtype, device=render.device)

    write_static_mask(tmp_path, "wide/frame.png", np.ones((128, 128), dtype=bool))
    mask = load_static_mask_for_evaluation(tmp_path, "wide/frame.png", (128, 128), (64, 64))
    assert mask.shape == (64, 64) and mask.all()
    render = torch.zeros((3, 64, 64), dtype=torch.float32)
    values = static_metric_values(render, render.clone(), torch.from_numpy(mask), ZeroLpips())
    assert values["PSNR_static"] == float("inf")
    assert values["SSIM_static"] == pytest.approx(1.0)
    assert values["LPIPS_static"] == pytest.approx(0.0)
    assert values["static_lpips_tile_count"] == 1

    bad_path = tmp_path / "masks" / "wide" / "bad.png"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((2, 2), 127, dtype=np.uint8)).save(bad_path)
    with pytest.raises(ValueError, match="binary"):
        load_static_mask_for_evaluation(tmp_path, "wide/bad.png", (2, 2), (2, 2))


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


def build_export_fixture(tmp_path, monkeypatch):
    """Stage a locked run so `export_artifacts` can be driven without a server."""

    yaml = pytest.importorskip("yaml")
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)

    colmap = tmp_path / "colmap"
    (colmap / "sparse" / "0").mkdir(parents=True)
    (colmap / "images").mkdir(parents=True)
    (colmap / "sparse" / "0" / "cameras.txt").write_text("# cameras\n", encoding="utf-8")
    (colmap / "sparse" / "0" / "images.txt").write_text(
        "# images\n1 1 0 0 0 0 0 0 1 frame_000.png\n\n", encoding="utf-8"
    )
    (colmap / "sparse" / "0" / "points3D.txt").write_text("# points\n", encoding="utf-8")
    (colmap / "metadata.json").write_text('{"records": []}', encoding="utf-8")
    (colmap / "images" / "frame_000.png").write_bytes(b"image")

    model = tmp_path / "model"
    iteration_dir = model / "point_cloud" / "iteration_90000"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "point_cloud_anchor.ply").write_bytes(b"anchors")
    with (model / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"model_params": {"source_path": str(colmap), "eval": False}, "optim_params": {}},
            handle,
        )

    uncertainty = tmp_path / "uncertainty"
    uncertainty.mkdir()
    np.save(uncertainty / "U.npy", np.zeros(4, dtype=np.float32))

    root = tmp_path / "run"
    root.mkdir()
    lock = {
        "octree_profile": "production",
        "uncertainty_profile": "baseline",
        "iteration": 90000,
        "model_path": str(model),
        "u_path": str(uncertainty / "U.npy"),
        "posterior_path": str(uncertainty / "anchor_posterior.npz"),
        "artifact_hashes": {"model_config": "hash"},
        "artifact_paths": {
            "model_config": str(model / "config.yaml"),
            "model_iteration": str(iteration_dir),
            "uncertainty": str(uncertainty / "U.npy"),
            "posterior": str(uncertainty / "anchor_posterior.npz"),
        },
    }

    runner_module = load_runner_module()
    experiment = object.__new__(runner_module.Experiment)
    with (repo_root / "configs" / "experiments" / "uncertainty-evaluation.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        experiment.config = yaml.safe_load(handle)
    experiment.root = root
    experiment.args = SimpleNamespace(dry_run=False)
    experiment.metadata_path = colmap / "metadata.json"
    experiment.verify_lock = lambda: lock
    # The colored-anchor ply is produced by a container command; skip it here.
    experiment.runner = SimpleNamespace(run=lambda *args, **kwargs: None)
    return experiment, colmap


def test_export_ships_colmap_cameras_and_a_portable_source_path(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    from vbogs.octree_config import load_octree_config_for_model

    experiment, colmap = build_export_fixture(tmp_path, monkeypatch)
    experiment.export_artifacts()

    export = experiment.root / "export"
    # The viewer resolves a relative source_path against --model-path, so the
    # patched config must land on the COLMAP tree shipped inside the export.
    resolved = Path(load_octree_config_for_model(export / "splat")["model_params"]["source_path"])
    assert resolved == (export / "prepared").resolve()
    assert (resolved / "sparse" / "0" / "images.txt").is_file()
    assert (resolved / "metadata.json").is_file()

    # Source images are not copied: the viewer uses metadata-only camera loading.
    assert not (resolved / "images").exists()

    # The unmodified config is what `model_config` in the lock hashes.
    with (export / "splat" / "original_config.yaml").open("r", encoding="utf-8") as handle:
        original = yaml.safe_load(handle)
    assert original["model_params"]["source_path"] == str(colmap)

    assert (export / "VIEWER_COMMANDS.md").is_file()
    assert "--camera-source train" in (export / "VIEWER_COMMANDS.md").read_text(encoding="utf-8")

    with (export / "export_manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["local_viewer"]["enabled"] is True
    assert manifest["local_viewer"]["config_source_path"] == "../prepared"


def test_export_without_prepared_colmap_leaves_the_config_untouched(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")

    experiment, colmap = build_export_fixture(tmp_path, monkeypatch)
    experiment.config["export"]["include_prepared_colmap"] = False
    experiment.export_artifacts()

    export = experiment.root / "export"
    assert not (export / "prepared").exists()
    assert not (export / "VIEWER_COMMANDS.md").exists()
    with (export / "splat" / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["model_params"]["source_path"] == str(colmap)
