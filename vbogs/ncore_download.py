"""Reusable NVIDIA PhysicalAI AV NCore Hugging Face download helpers.

The helpers intentionally take credentials as call arguments.  Callers must not
put a token in a command line, a log message, or a persisted job record.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
except Exception:  # pragma: no cover - optional runtime dependency
    HfApi = None
    HfHubHTTPError = Exception
    hf_hub_download = None


DEFAULT_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles-NCore"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_NCORE_ROOT = Path("/workspace/VBOGS/data/NVIDIA-PhysicalAI-AV-NCore")
CHUNK_SIZE = 1024 * 1024
Progress = Callable[[str], None]


class NCoreDownloadError(RuntimeError):
    """A safe, user-facing NCore download error."""


def _emit(progress: Progress | None, message: str) -> None:
    if progress:
        progress(message)


def auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_url(repo_id: str, revision: str, remote_path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{urllib.parse.quote(remote_path)}"


def hf_api_tree_url(repo_id: str, revision: str, path: str | None = None) -> str:
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{urllib.parse.quote(revision, safe='')}"
    if path:
        url += "/" + urllib.parse.quote(path)
    return url + "?recursive=true"


def list_repo_files(repo_id: str, revision: str, token: str, path: str | None = None) -> list[str]:
    """Return the complete repository or subtree index, following pagination."""

    try:
        if HfApi is not None:
            files = HfApi().list_repo_files(
                repo_id=repo_id, revision=revision, repo_type=DEFAULT_REPO_TYPE, token=token,
            )
            if path:
                prefix = path.rstrip("/") + "/"
                return [file_path for file_path in files if file_path.startswith(prefix)]
            return files

        paths: list[str] = []
        url: str | None = hf_api_tree_url(repo_id, revision, path)
        while url:
            request = urllib.request.Request(url, headers=auth_headers(token))
            with urllib.request.urlopen(request) as response:
                payload = json.loads(response.read().decode("utf-8"))
                link_header = response.headers.get("link", "")
            if not isinstance(payload, list):
                raise NCoreDownloadError("Unexpected Hugging Face catalog response.")
            paths.extend(item["path"] for item in payload if isinstance(item, dict) and "path" in item)
            next_link = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
            url = next_link.group(1) if next_link else None
        return paths
    except (HfHubHTTPError, urllib.error.HTTPError) as exc:
        raise NCoreDownloadError(
            "Unable to access the NCore Hugging Face dataset. Check the backend token and license acceptance."
        ) from exc
    except urllib.error.URLError as exc:
        raise NCoreDownloadError("Unable to reach Hugging Face while listing NCore clips.") from exc


def discover_scene_ids(repo_files: list[str]) -> list[str]:
    scenes: set[str] = set()
    for remote_path in repo_files:
        parts = remote_path.split("/")
        if len(parts) < 3 or parts[0] != "clips":
            continue
        scene_id = parts[1]
        if parts[2] == f"pai_{scene_id}.ncore4.zarr.itar":
            scenes.add(scene_id)
    return sorted(scenes)


def files_for_scene(scene_id: str, repo_files: list[str], mode: str = "full") -> list[str]:
    prefix = f"clips/{scene_id}/"
    if mode == "full":
        return sorted(path for path in repo_files if path.startswith(prefix))
    core_file = f"{prefix}pai_{scene_id}.ncore4.zarr.itar"
    if core_file not in repo_files:
        raise NCoreDownloadError(f"Missing required core file for clip {scene_id}.")
    metadata = f"{prefix}pai_{scene_id}.json"
    return [core_file, *([metadata] if metadata in repo_files else [])]


def scene_target_prefix(scene_id: str) -> Path:
    return Path("clips") / scene_id


def download_remote_file(
    repo_id: str, revision: str, token: str, remote_path: str, destination: Path, progress: Progress | None = None,
) -> Path:
    if hf_hub_download is not None:
        return Path(hf_hub_download(
            repo_id=repo_id, repo_type=DEFAULT_REPO_TYPE, revision=revision, token=token,
            filename=remote_path, local_dir=destination.parent, local_dir_use_symlinks=False,
        ))

    request = urllib.request.Request(hf_url(repo_id, revision, remote_path), headers=auth_headers(token))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total else None
        downloaded = 0
        next_report = 0
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                next_report = downloaded + 256 * 1024 * 1024
                if total_bytes:
                    _emit(progress, f"{remote_path}: {downloaded / 1e9:.2f} GB / {total_bytes / 1e9:.2f} GB")
                else:
                    _emit(progress, f"{remote_path}: {downloaded / 1e9:.2f} GB")
    return destination


def download_scene(
    scene_id: str,
    remote_files: list[str],
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
    token: str,
    ncore_root: Path = DEFAULT_NCORE_ROOT,
    force: bool = False,
    skip_existing: bool = True,
    dry_run: bool = False,
    progress: Progress | None = None,
) -> bool:
    """Download one complete clip, retaining valid components from prior attempts."""

    scene_dir = ncore_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    targets = [scene_dir / Path(path).relative_to(scene_target_prefix(scene_id)) for path in remote_files]
    if skip_existing and targets and all(path.exists() for path in targets):
        _emit(progress, f"Skipping {scene_id}; all reconstruction components already exist.")
        return True
    if dry_run:
        for remote_path in remote_files:
            _emit(progress, f"Would download {remote_path}")
        return True

    downloaded = 0
    with tempfile.TemporaryDirectory(prefix=f"ncore-{scene_id}-") as temporary:
        temp_dir = Path(temporary)
        for remote_path, target in zip(remote_files, targets):
            if target.exists() and not force:
                _emit(progress, f"Keeping existing component: {target.name}")
                continue
            _emit(progress, f"Downloading {remote_path}")
            local_path = download_remote_file(
                repo_id, revision, token, remote_path, temp_dir / Path(remote_path).name, progress,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            if force and target.exists():
                target.unlink()
            shutil.copy2(local_path, target)
            downloaded += 1
            _emit(progress, f"Downloaded {target.name}")
    return downloaded > 0 or all(path.exists() for path in targets)


def download_full_scene(
    scene_id: str, *, token: str, ncore_root: Path = DEFAULT_NCORE_ROOT,
    repo_id: str = DEFAULT_REPO_ID, revision: str = "main", progress: Progress | None = None,
) -> bool:
    """Fetch all NCore components required by VBOGS for one selected clip."""

    remote_files = list_repo_files(repo_id, revision, token, path=f"clips/{scene_id}")
    files = files_for_scene(scene_id, remote_files, mode="full")
    if not files:
        raise NCoreDownloadError(f"Clip {scene_id} was not found in the authorized catalog.")
    return download_scene(
        scene_id, files, repo_id=repo_id, revision=revision, token=token, ncore_root=ncore_root,
        skip_existing=True, progress=progress,
    )
