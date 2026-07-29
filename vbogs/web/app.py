"""FastAPI application for the VBOGS Web Experiment Console."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import json
import os
import re
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlparse, urlunparse

import yaml
from fastapi import Request, WebSocket

from vbogs.dataset_inventory import clips_to_json, list_dataset_clips
from vbogs.ncore_download import NCoreDownloadError
from vbogs.web.config import ConfigValidationError, load_presets, resolve_config
from vbogs.web.progress import project_run_progress
from vbogs.web.ncore_downloads import NCoreDownloadManager
from vbogs.web.scheduler import Scheduler, subprocess_runner
from vbogs.web.store import RunStore, TERMINAL_STATUSES, utc_now


STAGES = (
    "dynamic-mask", "prepare", "train", "stereo", "bucket", "fit", "inspect",
    "uncertainty", "map-viz", "render", "nbv", "nbv-viz", "bundle",
)
EXPERIMENT_STAGES = (
    "prepare", "octree-train", "points", "bucket", "uncertainty-fit", "test", "export", "report",
)
WORKFLOW_PIPELINE = "pipeline"
WORKFLOW_UNCERTAINTY = "uncertainty_evaluation"
EXPERIMENT_MODES = frozenset({"default", "smoke"})
DELETABLE_RUN_STATUSES = frozenset({"queued", "cancelled", "completed", "failed", "interrupted"})
PROXY_USER: ContextVar[str | None] = ContextVar("vbogs_proxy_user", default=None)


class WebError(ValueError):
    pass


@dataclass(frozen=True)
class ViewerInputs:
    """Renderer inputs proven to belong to one console run."""

    model_path: Path
    uncertainty_path: Path
    source: str


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def resolve_viewer_inputs(run: dict[str, Any]) -> ViewerInputs:
    """Resolve a renderable scene without accepting a browser supplied path."""

    if run.get("workflow", WORKFLOW_PIPELINE) == WORKFLOW_UNCERTAINTY:
        root = Path(run["output_path"]).resolve()
        model = root / "export" / "splat"
        uncertainty = root / "export" / "uncertainty" / "U.npy"
        if (model / "config.yaml").is_file() and uncertainty.is_file():
            return ViewerInputs(model, uncertainty, "uncertainty_experiment_export")
        raise WebError("Experiment export is not available yet (missing export/splat or export/uncertainty/U.npy)")

    output_root = (Path(run["output_path"]) / run["scene_id"]).resolve()
    portable_root = output_root / "local_viewer"
    portable_model = portable_root / "model"
    portable_uncertainty = portable_root / "uncertainty" / "U.npy"
    if (portable_model / "config.yaml").is_file() and portable_uncertainty.is_file():
        return ViewerInputs(portable_model, portable_uncertainty, "portable_export")

    workspace = Path(run["workspace_path"]).resolve()
    artifacts = workspace / "artifacts"
    record_path = artifacts / "train_run.json"
    uncertainty = artifacts / "m4" / str(run["scene_id"]) / "U.npy"
    if not record_path.is_file():
        raise WebError("Training output is not available yet (missing train_run.json)")
    if not uncertainty.is_file():
        raise WebError("Uncertainty is not available yet (missing U.npy)")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        model = Path(str(record["model_path"])).resolve()
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise WebError("Training output record is invalid") from exc
    if not _is_within(model, artifacts) or not (model / "config.yaml").is_file():
        raise WebError("Training output record points outside this run or lacks config.yaml")
    if not _is_within(uncertainty, artifacts):
        raise WebError("Uncertainty artifact is outside this run")
    return ViewerInputs(model, uncertainty, "run_workspace")


def viewer_readiness(run: dict[str, Any]) -> dict[str, Any]:
    """Return a public readiness diagnostic without leaking arbitrary paths."""

    try:
        inputs = resolve_viewer_inputs(run)
    except WebError as exc:
        return {"ready": False, "reason": str(exc), "source": None}
    if run["status"] != "completed":
        return {
            "ready": False,
            "reason": "The run must finish or be stopped after uncertainty before it can reserve a viewer GPU",
            "source": inputs.source,
        }
    return {"ready": True, "reason": None, "source": inputs.source}


def render_internal_url() -> str:
    value = os.environ.get("VBOGS_GUI_RENDER_INTERNAL_URL", "http://vbogs-vbgs-render:8070").rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
        raise WebError("VBOGS_GUI_RENDER_INTERNAL_URL must be an internal http(s) origin")
    return value


def render_websocket_url() -> str:
    parsed = urlparse(render_internal_url())
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, "/ws/render", "", "", ""))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def gpu_ids() -> tuple[str, ...]:
    values = tuple(item.strip() for item in os.environ.get("VBOGS_GUI_GPU_IDS", "0").split(",") if item.strip())
    if not values or any(not value.isdigit() for value in values) or len(set(values)) != len(values):
        raise RuntimeError("VBOGS_GUI_GPU_IDS must be a comma-separated unique list of GPU indexes")
    return values


def roles_for(user: str) -> set[str]:
    admins = {item.strip() for item in os.environ.get("VBOGS_GUI_ADMINS", "").split(",") if item.strip()}
    viewers = {item.strip() for item in os.environ.get("VBOGS_GUI_VIEWERS", "").split(",") if item.strip()}
    if user in admins:
        return {"viewer", "operator", "admin"}
    if not viewers or user in viewers:
        return {"viewer", "operator"}
    return {"viewer"}


def require_role(identity: dict[str, Any], role: str) -> None:
    if role not in identity["roles"]:
        raise PermissionError(f"The {role} role is required")


def stages_for(run: dict[str, Any]) -> tuple[str, ...]:
    return EXPERIMENT_STAGES if run.get("workflow", WORKFLOW_PIPELINE) == WORKFLOW_UNCERTAINTY else STAGES


def stage_preconditions(run: dict[str, Any], start_at: str) -> list[Path]:
    if run.get("workflow", WORKFLOW_PIPELINE) == WORKFLOW_UNCERTAINTY:
        return [Path(run["output_path"]) / "experiment_manifest.json"] if start_at != "prepare" else []
    root = Path(run["workspace_path"]) / "artifacts"
    scene = run["scene_id"]
    required: dict[str, list[Path]] = {
        "prepare": [],
        "train": [root / "colmap" / scene / "metadata.json"],
        "stereo": [root / "colmap" / scene / "metadata.json"],
        "bucket": [root / "points_world" / scene / "points_world.npz"],
        "fit": [root / "m4" / scene / "pts_by_anchor.npz"],
        "inspect": [root / "m4" / scene / "anchor_posterior.npz"],
        "uncertainty": [root / "m4" / scene / "anchor_posterior.npz"],
        "map-viz": [root / "m4" / scene / "U.npy"],
        "render": [root / "m4" / scene / "U.npy"],
        "nbv": [root / "m4" / scene / "U.npy"],
        "nbv-viz": [Path(run["output_path"]) / scene / "nbv" / "nbv_scores.json"],
        "bundle": [root / "m4" / scene / "U.npy"],
    }
    return required.get(start_at, [])


def _read_manifest(run: dict[str, Any]) -> dict[str, Any]:
    path = (
        Path(run["output_path"]) / "experiment_manifest.json"
        if run.get("workflow", WORKFLOW_PIPELINE) == WORKFLOW_UNCERTAINTY
        else Path(run["output_path"]) / run["scene_id"] / "run_manifest.json"
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def artifact_root(run: dict[str, Any]) -> Path:
    if run.get("workflow", WORKFLOW_PIPELINE) == WORKFLOW_UNCERTAINTY:
        return Path(run["output_path"]).resolve()
    return (Path(run["output_path"]) / run["scene_id"]).resolve()


def _artifact_index(run: dict[str, Any], limit: int = 250) -> list[str]:
    root = artifact_root(run)
    if not root.is_dir():
        return []
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
            if len(files) >= limit:
                break
    return files


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def uncertainty_results(run: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded, browser-safe summary of a completed experiment."""

    if run.get("workflow", WORKFLOW_PIPELINE) != WORKFLOW_UNCERTAINTY:
        raise WebError("This run is not an uncertainty evaluation")
    root = artifact_root(run)
    test = _json_object(root / "test" / "summary.json")
    octree = _json_object(root / "octree_selection.json")
    uncertainty = _json_object(root / "uncertainty_selection.json")
    metadata = _json_object(root / "export" / "uncertainty" / "uncertainty_metadata.json")
    plots = [
        name for name in (
            "plots/test_calibration_scatter.png", "plots/test_sparsification.png",
            "plots/test_calibration_scatter_legacy.png", "plots/test_sparsification_legacy.png",
            "plots/uncertainty_histogram.png",
        ) if (root / name).is_file()
    ]
    renders = [
        path.relative_to(root).as_posix() for path in sorted((root / "test" / "renders").glob("*"))
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ][:24] if (root / "test" / "renders").is_dir() else []
    groups: dict[str, Any] = {}
    if test and isinstance(test.get("groups"), dict):
        for name in ("primary", "all"):
            group = test["groups"].get(name)
            if not isinstance(group, dict):
                continue
            groups[name] = {
                "view_count": group.get("view_count"), "metrics": group.get("metrics"),
                "calibration": group.get("calibration"), "calibration_observed": group.get("calibration_observed"),
                "unobserved_fraction": group.get("unobserved_fraction"),
            }
    return {
        "ready": bool(test), "groups": groups,
        "octree_selection": octree.get("selected") if octree else None,
        "uncertainty_selection": uncertainty.get("selected") if uncertainty else None,
        "uncertainty_metadata": metadata, "plots": plots, "renders": renders,
        "report": "report.md" if (root / "report.md").is_file() else None,
        "export_ready": (root / "export" / "splat" / "config.yaml").is_file()
        and (root / "export" / "uncertainty" / "U.npy").is_file(),
    }


def run_storage_paths(
    run: dict[str, Any], *, data_root: Path, output_root: Path, project_root: Path | None = None,
    experiment_octree_root: Path | None = None,
) -> tuple[Path, ...]:
    """Return the two run-owned storage roots after strict path validation."""

    run_id = str(run["id"])
    expected_workspace = data_root.resolve() / "runs" / run_id
    workspace = Path(str(run["workspace_path"]))
    output = Path(str(run["output_path"]))
    if run.get("workflow", WORKFLOW_PIPELINE) == WORKFLOW_UNCERTAINTY:
        if project_root is None or experiment_octree_root is None:
            raise WebError("Experiment storage validation requires configured roots")
        expected_output = project_root.resolve() / "outputs" / "experiments" / "uncertainty-evaluation" / run["dataset"] / run["scene_id"] / run_id
        expected_work = project_root.resolve() / "data" / "experiments" / "uncertainty-evaluation" / run["dataset"] / run["scene_id"] / run_id
        expected_octree = experiment_octree_root.resolve() / run["dataset"] / run["scene_id"] / run_id
        if workspace.resolve() != expected_workspace or output.resolve() != expected_output:
            raise WebError("Run storage paths do not match this GUI uncertainty experiment")
        paths = (workspace, output, expected_work, expected_octree)
    else:
        expected_output = output_root.resolve() / run_id
        if workspace.resolve() != expected_workspace or output.resolve() != expected_output:
            raise WebError("Run storage paths do not match this GUI run")
        paths = (workspace, output)
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(run[key])) for key in ("id", "dataset", "scene_id")):
        raise WebError("Run storage identifiers are unsafe")
    for path in paths:
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise WebError("Run storage root is not a regular directory")
    return paths


def remove_run_storage(
    run: dict[str, Any], *, data_root: Path, output_root: Path, project_root: Path | None = None,
    experiment_octree_root: Path | None = None,
) -> None:
    """Delete only the workspace and output directories owned by one GUI run."""

    for path in run_storage_paths(
        run, data_root=data_root, output_root=output_root, project_root=project_root,
        experiment_octree_root=experiment_octree_root,
    ):
        if path.exists():
            shutil.rmtree(path)


def _render_container() -> str:
    project = os.environ.get("VBOGS_COMPOSE_PROJECT", "")
    filters = ["--filter", "label=com.docker.compose.service=vbogs-vbgs-render", "--filter", "status=running"]
    if project:
        filters.extend(["--filter", f"label=com.docker.compose.project={project}"])
    result = subprocess.run(["docker", "ps", "-q", *filters], check=True, capture_output=True, text=True)
    ids = [value.strip() for value in result.stdout.splitlines() if value.strip()]
    if len(ids) != 1:
        raise WebError("Expected exactly one running vbogs-vbgs-render container")
    return ids[0]


def _start_shared_viewer(run: dict[str, Any], gpu: str, inputs: ViewerInputs | None = None) -> None:
    inputs = inputs or resolve_viewer_inputs(run)
    container = _render_container()
    # The run paths are generated by this service and scene ids are validated at
    # submission. No browser-provided command fragment is ever interpolated.
    subprocess.run(
        ["docker", "exec", "-w", "/workspace/VBOGS", container, "sh", "-lc", "pkill -f '[s]cripts/view_octree_anygs.py' || true"],
        check=True,
        capture_output=True,
        text=True,
    )
    command = "exec " + shlex.join([
        "python", "scripts/view_octree_anygs.py", "--model-path", str(inputs.model_path),
        "--u-path", str(inputs.uncertainty_path), "--resolution", "4", "--camera-source", "train",
    ])
    subprocess.run(
        ["docker", "exec", "-d", "-w", "/workspace/VBOGS", "-e", f"CUDA_VISIBLE_DEVICES={gpu}", container, "sh", "-lc", command],
        check=True,
        capture_output=True,
        text=True,
    )


def _stop_shared_viewer() -> None:
    container = _render_container()
    subprocess.run(
        ["docker", "exec", "-w", "/workspace/VBOGS", container, "sh", "-lc", "pkill -f '[s]cripts/view_octree_anygs.py' || true"],
        check=True,
        capture_output=True,
        text=True,
    )


async def proxy_renderer_json(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    """Forward a small JSON request to the renderer on the private Compose network."""

    try:
        import httpx

        async with httpx.AsyncClient(base_url=render_internal_url(), timeout=90.0) as client:
            response = await client.request(method, path, json=payload)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        raise WebError(f"Renderer request failed: {exc}") from exc


def create_app(*, root: Path | None = None, store_path: Path | None = None):
    from fastapi import FastAPI, Header, HTTPException, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    project_root = (root or repo_root()).resolve()
    data_root = Path(os.environ.get("VBOGS_GUI_DATA_ROOT", project_root / "data" / "gui")).resolve()
    output_root = Path(os.environ.get("VBOGS_GUI_OUTPUT_ROOT", project_root / "outputs" / "gui" / "runs")).resolve()
    experiment_octree_root = Path(
        os.environ.get("VBOGS_EXPERIMENT_OCTREE_ROOT", "/data/OCTREE-ANYGS/uncertainty-evaluation")
    ).resolve()
    db_path = store_path or data_root / "control.sqlite3"
    store = RunStore(db_path)
    presets = load_presets(project_root / "configs" / "gui" / "presets", project_root)
    scheduler = Scheduler(store, gpu_ids(), subprocess_runner)
    ncore_downloads = NCoreDownloadManager(store)
    app = FastAPI(title="VBOGS Web Experiment Console")
    app.state.store = store
    app.state.scheduler = scheduler
    app.state.ncore_downloads = ncore_downloads
    app.state.presets = presets
    app.state.project_root = project_root
    app.state.data_root = data_root
    app.state.output_root = output_root
    app.state.experiment_octree_root = experiment_octree_root
    app.state.viewer_lock = asyncio.Lock()

    @app.middleware("http")
    async def proxy_identity(request, call_next):
        PROXY_USER.set(request.headers.get(os.environ.get("VBOGS_AUTH_USER_HEADER", "X-Forwarded-User")))
        return await call_next(request)

    static_dir = Path(os.environ.get("VBOGS_GUI_STATIC_DIR", project_root / "web" / "dist"))
    if static_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

    @app.on_event("startup")
    async def startup() -> None:
        scheduler.start()
        ncore_downloads.start()

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await ncore_downloads.stop()
        await scheduler.stop()

    def identity(x_forwarded_user: str | None = Header(default=None)) -> dict[str, Any]:
        user = PROXY_USER.get() or x_forwarded_user
        if not user:
            raise HTTPException(status_code=401, detail="Authenticated proxy identity is required")
        return {"user": user, "roles": sorted(roles_for(user))}

    def websocket_identity(websocket: WebSocket) -> dict[str, Any] | None:
        user = websocket.headers.get(os.environ.get("VBOGS_AUTH_USER_HEADER", "X-Forwarded-User"))
        if not user:
            return None
        return {"user": user, "roles": sorted(roles_for(user))}

    def run_or_404(run_id: str) -> dict[str, Any]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    def authorize_owner(identity_value: dict[str, Any], run: dict[str, Any]) -> None:
        if "admin" not in identity_value["roles"] and run["owner"] != identity_value["user"]:
            raise HTTPException(status_code=403, detail="Only the owning operator or an admin may change this run")

    @app.get("/api/me")
    async def me(x_forwarded_user: str | None = Header(default=None)):
        return identity(x_forwarded_user)

    @app.get("/api/presets")
    async def get_presets(x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        return [preset.public() for preset in presets.values()]

    @app.get("/api/datasets")
    async def datasets(x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        clips = list_dataset_clips(dataset_name="all")
        return json.loads(clips_to_json(clips))

    @app.get("/api/ncore/catalog")
    async def ncore_catalog(
        query: str = "", limit: int = 100, x_forwarded_user: str | None = Header(default=None),
        hf_token: str | None = Header(default=None, alias="X-VBOGS-HF-Token"),
    ):
        principal = identity(x_forwarded_user)
        try:
            require_role(principal, "operator")
            scene_ids = await ncore_downloads.catalog(token=hf_token, query=query, limit=limit)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (ValueError, NCoreDownloadError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        local = {clip.scene_id: clip.status for clip in list_dataset_clips(dataset_name="nvidia_ncore")}
        return {"clips": [{"scene_id": scene_id, "status": local.get(scene_id, "missing")} for scene_id in scene_ids]}

    @app.get("/api/ncore/downloads")
    async def ncore_download_list(x_forwarded_user: str | None = Header(default=None)):
        principal = identity(x_forwarded_user)
        try:
            require_role(principal, "operator")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return store.list_downloads()

    @app.post("/api/ncore/downloads")
    async def ncore_download_create(request: Request, x_forwarded_user: str | None = Header(default=None)):
        principal = identity(x_forwarded_user)
        try:
            require_role(principal, "operator")
            payload = await request.json()
            if not isinstance(payload, dict):
                raise WebError("Expected a JSON object")
            scene_id = str(payload["scene_id"])
            hf_token = request.headers.get("X-VBOGS-HF-Token")
            catalog = await ncore_downloads.catalog(token=hf_token, query=scene_id, limit=2)
            if scene_id not in catalog:
                raise WebError("Select a clip from the authorized NCore catalog")
            local = {clip.scene_id: clip.status for clip in list_dataset_clips(dataset_name="nvidia_ncore")}
            if local.get(scene_id) == "ready":
                raise WebError("This clip is already ready for the NCore reconstruction pipeline")
            return JSONResponse(
                ncore_downloads.enqueue(scene_id=scene_id, owner=principal["user"], token=hf_token), status_code=201,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (KeyError, ValueError, WebError, NCoreDownloadError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/ncore/downloads/{download_id}/log")
    async def ncore_download_log(download_id: str, x_forwarded_user: str | None = Header(default=None)):
        principal = identity(x_forwarded_user)
        try:
            require_role(principal, "operator")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if store.get_download(download_id) is None:
            raise HTTPException(status_code=404, detail="NCore download not found")
        return {"events": store.download_events(download_id)}

    @app.get("/api/slots")
    async def slots(x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        return scheduler.slots()

    @app.get("/api/runs")
    async def runs(scope: str = "all", x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        scopes = {
            "all": None,
            "active": ("queued", "starting", "running", "cancelling"),
            "completed": ("completed",),
            "recoverable": ("failed", "cancelled", "interrupted"),
        }
        if scope not in scopes:
            raise HTTPException(status_code=422, detail="scope must be one of: all, active, completed, recoverable")
        return store.list_runs(statuses=scopes[scope])

    @app.post("/api/runs")
    async def create_run(request: Request, x_forwarded_user: str | None = Header(default=None)):
        principal = identity(x_forwarded_user)
        try:
            require_role(principal, "operator")
            payload = await request.json()
            if not isinstance(payload, dict):
                raise WebError("Expected a JSON object")
            dataset = str(payload["dataset"])
            scene_id = str(payload["scene_id"])
            workflow = str(payload.get("workflow", WORKFLOW_PIPELINE))
            if workflow not in {WORKFLOW_PIPELINE, WORKFLOW_UNCERTAINTY}:
                raise WebError("workflow must be pipeline or uncertainty_evaluation")
            workflow_stages = EXPERIMENT_STAGES if workflow == WORKFLOW_UNCERTAINTY else STAGES
            start_at = str(payload.get("start_at", "prepare"))
            stop_after = str(payload.get("stop_after", "report" if workflow == WORKFLOW_UNCERTAINTY else "bundle"))
            if start_at not in workflow_stages or stop_after not in workflow_stages or workflow_stages.index(start_at) > workflow_stages.index(stop_after):
                raise WebError("Invalid pipeline stage slice")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", scene_id) or scene_id in {".", ".."}:
                raise WebError("Scene identifier contains unsupported characters")
            if workflow == WORKFLOW_UNCERTAINTY:
                experiment_mode = str(payload.get("experiment_mode", "default"))
                if experiment_mode not in EXPERIMENT_MODES:
                    raise WebError("experiment_mode must be default or smoke")
                if payload.get("overrides") or payload.get("advanced_yaml"):
                    raise WebError("Uncertainty evaluations do not accept browser configuration overrides")
                preset_name = "uncertainty-evaluation"
                config = None
            else:
                preset_name = str(payload["preset"])
                experiment_mode = None
                preset = presets.get(preset_name)
                if preset is None:
                    raise WebError("Unknown preset")
                config = resolve_config(
                    preset, repo_root=project_root, dataset=dataset, scene_id=scene_id,
                    overrides=payload.get("overrides"), advanced_yaml=payload.get("advanced_yaml"),
                )
        except (KeyError, WebError, ConfigValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        workspace = data_root / "runs" / run_id
        output = (
            project_root / "outputs" / "experiments" / "uncertainty-evaluation" / dataset / scene_id / run_id
            if workflow == WORKFLOW_UNCERTAINTY else output_root / run_id
        )
        workspace.mkdir(parents=True, exist_ok=False)
        config_path = workspace / ("uncertainty_evaluation.yaml" if workflow == WORKFLOW_UNCERTAINTY else "resolved_config.yaml")
        if workflow == WORKFLOW_UNCERTAINTY:
            shutil.copy2(project_root / "configs" / "experiments" / "uncertainty-evaluation.yaml", config_path)
            config_path.chmod(0o444)
            command = ["scripts/uncertainty-evaluation", "--config", str(config_path), "--dataset-name", dataset, "--scene-id", scene_id, "--run-id", run_id]
            if experiment_mode == "smoke":
                command.append("--smoke")
        else:
            config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
            command = ["scripts/run_pipeline.sh", "--config", str(config_path), "--start-at", start_at, "--stop-after", stop_after]
        request_record = {"submitted_at": utc_now(), "submitted_by": principal["user"], "request": payload}
        (workspace / "request.json").write_text(json.dumps(request_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        run = store.create_run({
            "id": run_id, "owner": principal["user"], "dataset": dataset, "scene_id": scene_id,
            "preset": preset_name, "workflow": workflow, "experiment_mode": experiment_mode,
            "start_at": start_at, "stop_after": stop_after, "created_at": utc_now(),
            "config_path": str(config_path), "workspace_path": str(workspace), "output_path": str(output), "command": command,
        })
        scheduler.notify()
        return JSONResponse(run, status_code=201)

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str, x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        run = run_or_404(run_id)
        return {
            **run,
            "manifest": _read_manifest(run),
            "events": store.events(run_id),
            "artifacts": _artifact_index(run),
            "progress": project_run_progress(run),
            "results": uncertainty_results(run) if run.get("workflow", WORKFLOW_PIPELINE) == WORKFLOW_UNCERTAINTY else None,
        }

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel(run_id: str, x_forwarded_user: str | None = Header(default=None)):
        principal = identity(x_forwarded_user)
        run = run_or_404(run_id)
        authorize_owner(principal, run)
        if run["status"] in TERMINAL_STATUSES:
            raise HTTPException(status_code=409, detail="Run is already terminal")
        await scheduler.cancel(run_id)
        return run_or_404(run_id)

    @app.post("/api/runs/{run_id}/resume")
    async def resume(run_id: str, request: Request, x_forwarded_user: str | None = Header(default=None)):
        principal = identity(x_forwarded_user)
        run = run_or_404(run_id)
        authorize_owner(principal, run)
        if run["status"] not in {"failed", "cancelled", "interrupted"}:
            raise HTTPException(status_code=409, detail="Only failed, cancelled, or interrupted runs may resume")
        payload = await request.json()
        start_at = str(payload.get("start_at", run["start_at"]))
        stop_after = str(payload.get("stop_after", run["stop_after"]))
        stages = stages_for(run)
        if start_at not in stages or stop_after not in stages or stages.index(start_at) > stages.index(stop_after):
            raise HTTPException(status_code=422, detail="Invalid pipeline stage slice")
        missing = [str(path) for path in stage_preconditions(run, start_at) if not path.exists()]
        if missing:
            raise HTTPException(status_code=422, detail={"message": "Missing resume artifacts", "paths": missing})
        cancel_file = Path(run["workspace_path"]) / "cancel.request"
        cancel_file.unlink(missing_ok=True)
        (Path(run["workspace_path"]) / "training_progress.json").unlink(missing_ok=True)
        resumed = store.requeue(run_id, start_at=start_at, stop_after=stop_after)
        scheduler.notify()
        return resumed

    @app.delete("/api/runs/{run_id}")
    async def delete_run(run_id: str, request: Request, x_forwarded_user: str | None = Header(default=None)):
        principal = identity(x_forwarded_user)
        run = run_or_404(run_id)
        authorize_owner(principal, run)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=422, detail="Expected a JSON confirmation object") from exc
        if not isinstance(payload, dict) or payload.get("confirm_run_id") != run_id:
            raise HTTPException(status_code=422, detail="Type the exact run ID to confirm deletion")
        if run["status"] not in DELETABLE_RUN_STATUSES:
            raise HTTPException(status_code=409, detail="Only queued, cancelled, completed, failed, or interrupted runs may be deleted")

        async with app.state.viewer_lock:
            run = run_or_404(run_id)
            if run["status"] not in DELETABLE_RUN_STATUSES:
                raise HTTPException(status_code=409, detail="Run status changed before deletion completed")
            try:
                run_storage_paths(
                    run, data_root=data_root, output_root=output_root, project_root=project_root,
                    experiment_octree_root=experiment_octree_root,
                )
            except WebError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            active_viewer = store.viewer()
            if active_viewer and active_viewer.get("run_id") == run_id:
                try:
                    _stop_shared_viewer()
                except (OSError, subprocess.CalledProcessError, WebError) as exc:
                    raise HTTPException(status_code=503, detail=f"Could not stop shared viewer: {exc}") from exc
                store.clear_viewer()
            try:
                remove_run_storage(
                    run, data_root=data_root, output_root=output_root, project_root=project_root,
                    experiment_octree_root=experiment_octree_root,
                )
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Could not delete run storage: {exc}") from exc
            deleted = store.delete_run(run_id, allowed_statuses=DELETABLE_RUN_STATUSES)
        if deleted is None:
            raise HTTPException(status_code=409, detail="Run status changed before deletion completed")
        scheduler.notify()
        return {"id": run_id, "deleted": True}

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, after: int = 0, x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        run_or_404(run_id)
        return store.events(run_id, after)

    @app.get("/api/runs/{run_id}/events/stream")
    async def stream_events(run_id: str, x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        run = run_or_404(run_id)
        event_path = Path(run["workspace_path"]) / "pipeline.events.jsonl"
        async def stream() -> AsyncIterator[str]:
            sequence = 0
            byte_offset = 0
            progress_payload: str | None = None
            while True:
                for event in store.events(run_id, sequence):
                    sequence = event["sequence"]
                    yield f"id: db-{sequence}\nevent: {event['type']}\ndata: {json.dumps(event)}\n\n"
                if event_path.exists():
                    with event_path.open("r", encoding="utf-8") as handle:
                        handle.seek(byte_offset)
                        for line in handle:
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            yield f"event: pipeline\ndata: {json.dumps(event)}\n\n"
                        byte_offset = handle.tell()
                latest = store.get_run(run_id)
                if latest:
                    latest_progress = json.dumps(project_run_progress(latest), sort_keys=True)
                    if latest_progress != progress_payload:
                        progress_payload = latest_progress
                        yield f"event: progress\ndata: {latest_progress}\n\n"
                if latest and latest["status"] in TERMINAL_STATUSES:
                    yield "event: terminal\ndata: {}\n\n"
                    return
                await asyncio.sleep(0.75)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/runs/{run_id}/log")
    async def log(run_id: str, tail: int = 200, x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        run = run_or_404(run_id)
        path = Path(run["workspace_path"]) / "pipeline.log"
        if not path.exists():
            return {"lines": []}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"lines": lines[-max(1, min(tail, 5000)):]} 

    @app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
    async def artifact(run_id: str, artifact_path: str, x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        run = run_or_404(run_id)
        root_path = artifact_root(run)
        candidate = (root_path / artifact_path).resolve()
        if root_path not in candidate.parents or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(candidate)

    @app.get("/api/runs/{run_id}/results")
    async def results(run_id: str, x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        try:
            return uncertainty_results(run_or_404(run_id))
        except WebError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/compare")
    async def compare(request: Request, x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        payload = await request.json()
        ids = payload.get("run_ids") if isinstance(payload, dict) else None
        if not isinstance(ids, list) or len(ids) != 2 or ids[0] == ids[1]:
            raise HTTPException(status_code=422, detail="Select exactly two different run IDs")
        left, right = (run_or_404(str(run_id)) for run_id in ids)
        return {"runs": [{**run, "manifest": _read_manifest(run), "artifacts": _artifact_index(run)} for run in (left, right)]}

    @app.get("/api/runs/{run_id}/viewer-readiness")
    async def viewer_readiness_endpoint(run_id: str, x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        return viewer_readiness(run_or_404(run_id))

    def active_viewer_or_409() -> dict[str, Any]:
        value = store.viewer()
        if not value or not value.get("run_id") or value.get("status") != "active":
            raise HTTPException(status_code=409, detail="No active shared viewer session")
        return value

    @app.get("/api/viewer")
    async def viewer(x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        return store.viewer() or {"run_id": None, "status": "idle", "revision": 0}

    @app.post("/api/viewer")
    async def set_viewer(request: Request, x_forwarded_user: str | None = Header(default=None)):
        principal = identity(x_forwarded_user)
        try:
            require_role(principal, "operator")
            payload = await request.json()
            if not isinstance(payload, dict):
                raise WebError("Expected a JSON object")
            run = run_or_404(str(payload["run_id"]))
            readiness = viewer_readiness(run)
            if not readiness["ready"]:
                raise WebError(str(readiness["reason"]))
            inputs = resolve_viewer_inputs(run)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (KeyError, WebError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        async with app.state.viewer_lock:
            active = store.viewer()
            replacement = bool(active and active.get("run_id") and active.get("run_id") != run["id"])
            if replacement and not bool(payload.get("confirm_replace", False)):
                raise HTTPException(
                    status_code=409,
                    detail={"message": "Confirm replacement of the active shared viewer", "active_viewer": active},
                )
            available = [slot["gpu_id"] for slot in scheduler.slots() if not slot["run_id"]]
            if not available:
                raise HTTPException(status_code=409, detail="All configured GPUs are occupied by pipeline runs")
            gpu = str(active["gpu_id"]) if replacement and active.get("gpu_id") in available else str(available[0])
            try:
                _start_shared_viewer(run, gpu, inputs)
            except (OSError, subprocess.CalledProcessError, WebError) as exc:
                raise HTTPException(status_code=503, detail=f"Could not start shared viewer: {exc}") from exc
            state = store.set_viewer(run["id"], gpu, principal["user"])
            store.add_event(run["id"], "viewer_loaded", {"gpu_id": gpu, "source": inputs.source})
            return state

    @app.delete("/api/viewer")
    async def stop_viewer(x_forwarded_user: str | None = Header(default=None)):
        principal = identity(x_forwarded_user)
        try:
            require_role(principal, "operator")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        async with app.state.viewer_lock:
            active = store.viewer()
            if not active or not active.get("run_id"):
                return store.clear_viewer()
            try:
                _stop_shared_viewer()
            except (OSError, subprocess.CalledProcessError, WebError) as exc:
                raise HTTPException(status_code=503, detail=f"Could not stop shared viewer: {exc}") from exc
            state = store.clear_viewer()
            store.add_event(str(active["run_id"]), "viewer_stopped", {"by": principal["user"]})
            return state

    @app.get("/api/viewer/metadata")
    async def viewer_metadata(x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        active_viewer_or_409()
        try:
            return await proxy_renderer_json("GET", "/api/metadata")
        except WebError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/viewer/cameras")
    async def viewer_cameras(x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        active_viewer_or_409()
        try:
            return await proxy_renderer_json("GET", "/api/cameras")
        except WebError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/viewer/render")
    async def viewer_render(request: Request, x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        active_viewer_or_409()
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="Expected a JSON render request")
        try:
            return await proxy_renderer_json("POST", "/api/render", payload)
        except WebError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/viewer/rendered-anchors")
    async def viewer_rendered_anchors(request: Request, x_forwarded_user: str | None = Header(default=None)):
        identity(x_forwarded_user)
        active_viewer_or_409()
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="Expected a JSON rendered-anchor request")
        try:
            return await proxy_renderer_json("POST", "/api/rendered-anchors", payload)
        except WebError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.websocket("/api/viewer/ws/render")
    async def viewer_render_websocket(websocket: WebSocket):
        socket_identity = websocket_identity(websocket)
        if socket_identity is None:
            await websocket.close(code=4401)
            return
        try:
            active_viewer_or_409()
        except HTTPException:
            await websocket.close(code=4409)
            return
        await websocket.accept()
        try:
            import websockets

            async with websockets.connect(render_websocket_url(), max_size=None) as renderer_socket:
                async def browser_to_renderer() -> None:
                    while True:
                        message = await websocket.receive()
                        if message.get("type") == "websocket.disconnect":
                            return
                        if message.get("text") is not None:
                            await renderer_socket.send(message["text"])

                async def renderer_to_browser() -> None:
                    async for message in renderer_socket:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)

                left = asyncio.create_task(browser_to_renderer())
                right = asyncio.create_task(renderer_to_browser())
                done, pending = await asyncio.wait({left, right}, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    task.result()
        except WebSocketDisconnect:
            return
        except Exception:
            await websocket.close(code=1011)

    @app.get("/")
    async def index():
        index_path = static_dir / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)
        return JSONResponse({"service": "vbogs-web", "detail": "Frontend assets were not built."})

    @app.get("/{spa_path:path}")
    async def spa(spa_path: str):
        index_path = static_dir / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)
        return JSONResponse({"service": "vbogs-web", "detail": "Frontend assets were not built."})

    return app
