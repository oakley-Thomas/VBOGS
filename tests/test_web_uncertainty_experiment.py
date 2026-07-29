import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from vbogs.web.app import WebError, resolve_viewer_inputs, run_storage_paths, uncertainty_results
from vbogs.web.progress import project_run_progress


def experiment_run(tmp_path: Path) -> dict:
    project = tmp_path / "project"
    run_id = "run-experiment-123"
    return {
        "id": run_id,
        "workflow": "uncertainty_evaluation",
        "experiment_mode": "smoke",
        "status": "completed",
        "dataset": "kitti360",
        "scene_id": "scene",
        "start_at": "prepare",
        "stop_after": "report",
        "workspace_path": str(project / "data" / "gui" / "runs" / run_id),
        "output_path": str(project / "outputs" / "experiments" / "uncertainty-evaluation" / "kitti360" / "scene" / run_id),
    }


def test_experiment_export_is_viewer_ready_and_results_are_bounded(tmp_path):
    run = experiment_run(tmp_path)
    root = Path(run["output_path"])
    (root / "export" / "splat").mkdir(parents=True)
    (root / "export" / "splat" / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (root / "export" / "uncertainty").mkdir()
    (root / "export" / "uncertainty" / "U.npy").write_bytes(b"u")
    (root / "test" / "renders").mkdir(parents=True)
    for index in range(30):
        (root / "test" / "renders" / f"{index}.png").write_bytes(b"image")
    (root / "test" / "summary.json").write_text(json.dumps({"groups": {"primary": {"view_count": 2, "metrics": {"PSNR": {"mean": 20.0}}}}}), encoding="utf-8")
    (root / "octree_selection.json").write_text(json.dumps({"selected": {"profile": "production", "iteration": 90000}}), encoding="utf-8")
    (root / "uncertainty_selection.json").write_text(json.dumps({"selected": {"profile": "baseline"}}), encoding="utf-8")

    viewer = resolve_viewer_inputs(run)
    results = uncertainty_results(run)

    assert viewer.source == "uncertainty_experiment_export"
    assert results["ready"] is True
    assert results["octree_selection"]["profile"] == "production"
    assert len(results["renders"]) == 24


def test_experiment_storage_paths_must_match_the_canonical_roots(tmp_path):
    run = experiment_run(tmp_path)
    project = tmp_path / "project"
    paths = run_storage_paths(
        run,
        data_root=project / "data" / "gui",
        output_root=project / "outputs" / "gui" / "runs",
        project_root=project,
        experiment_octree_root=tmp_path / "octree",
    )
    assert len(paths) == 4
    run["output_path"] = str(tmp_path / "outside")
    with pytest.raises(WebError, match="uncertainty experiment"):
        run_storage_paths(
            run,
            data_root=project / "data" / "gui",
            output_root=project / "outputs" / "gui" / "runs",
            project_root=project,
            experiment_octree_root=tmp_path / "octree",
        )


def test_experiment_progress_uses_experiment_stage_names(tmp_path):
    run = experiment_run(tmp_path)
    workspace = Path(run["workspace_path"])
    workspace.mkdir(parents=True)
    (workspace / "pipeline.events.jsonl").write_text(
        '{"type":"run_started","stages":["prepare","octree-train"]}\n'
        '{"type":"stage_started","stage":"prepare"}\n'
        '{"type":"stage_completed","stage":"prepare"}\n'
        '{"type":"stage_started","stage":"octree-train"}\n',
        encoding="utf-8",
    )
    run["status"] = "running"
    progress = project_run_progress(run)
    assert progress["current_stage"] == {"name": "octree-train", "index": 2, "total": 2}
    assert progress["overall"]["percent"] == 50.0
