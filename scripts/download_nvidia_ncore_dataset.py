#!/usr/bin/env python3

"""Download NVIDIA PhysicalAI AV NCore clips from Hugging Face.

Examples:
  python scripts/download_nvidia_ncore_dataset.py --all --max-scenes 5
  python scripts/download_nvidia_ncore_dataset.py --scene-id pai_123,pai_456
  python scripts/download_nvidia_ncore_dataset.py --scene-id-file clips.txt --mode full

Downloaded clips are written under:
  /workspace/VBOGS/data/NVIDIA-PhysicalAI-AV-NCore/<scene-id>/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

try:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
except Exception:  # pragma: no cover - env-specific
    HfApi = None
    HfHubHTTPError = Exception
    hf_hub_download = None


DEFAULT_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles-NCore"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_NCORE_ROOT = Path("/workspace/VBOGS/data/NVIDIA-PhysicalAI-AV-NCore")
DEFAULT_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGING_FACE_TOKEN")
CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repository id.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Repository revision or branch (default: main).",
    )
    parser.add_argument(
        "--ncore-root",
        type=Path,
        default=DEFAULT_NCORE_ROOT,
        help="Destination root for downloaded clips.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (defaults to HF_TOKEN/HUGGING_FACE_HUB_TOKEN).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available clips.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        default=None,
        help="Scene id to download (repeatable, comma-separated values supported).",
    )
    parser.add_argument(
        "--scene-id-file",
        type=Path,
        default=None,
        help="Text file with one scene id per line.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Limit scenes when using --all (0 = unlimited).",
    )
    parser.add_argument(
        "--mode",
        choices=("core-only", "full"),
        default="core-only",
        help=(
            "core-only: download only the core NCore + metadata file; "
            "full: download everything under clips/<scene-id>/"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scene if required core file already exists.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing scene files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without downloading files.",
    )
    return parser.parse_args()


def parse_scene_tokens(values: Iterable[str]) -> list[str]:
    ids: list[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                ids.append(token)
    return ids


def parse_scene_file(path: Path) -> list[str]:
    ids: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        ids.extend(parse_scene_tokens([raw]))
    return ids


def resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    for env_var in DEFAULT_TOKEN_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            return value
    return None


def dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def discover_scene_ids(repo_files: list[str]) -> list[str]:
    scenes: list[str] = []
    seen: set[str] = set()
    for path in repo_files:
        parts = path.split("/")
        if len(parts) < 2 or parts[0] != "clips":
            continue
        scene_id = parts[1]
        if len(parts) >= 3:
            core_file = f"pai_{scene_id}.ncore4.zarr.itar"
            if parts[2] != core_file:
                continue
        if scene_id in seen:
            continue
        seen.add(scene_id)
        scenes.append(scene_id)
    scenes.sort()
    return scenes


def files_for_scene(scene_id: str, repo_files: list[str], mode: str) -> list[str]:
    prefix = f"clips/{scene_id}/"
    if mode == "full":
        return [path for path in repo_files if path.startswith(prefix)]

    core_file = f"{prefix}pai_{scene_id}.ncore4.zarr.itar"
    if core_file not in repo_files:
        raise ValueError(f"Missing required core file in repo: {core_file}")
    files = [core_file]
    meta_file = f"{prefix}pai_{scene_id}.json"
    if meta_file in repo_files:
        files.append(meta_file)
    return files


def scene_target_prefix(scene_id: str) -> Path:
    return Path("clips") / scene_id


def copy_downloaded_file(local_path: Path, scene_dir: Path, scene_id: str, remote_path: str) -> Path:
    relative_remote = Path(remote_path).relative_to(scene_target_prefix(scene_id))
    destination = scene_dir / relative_remote
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, destination)
    return destination


def auth_headers(token: str | None) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def hf_url(repo_id: str, revision: str, remote_path: str) -> str:
    encoded_path = urllib.parse.quote(remote_path)
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{encoded_path}"


def hf_api_tree_url(repo_id: str, revision: str, path: str | None = None) -> str:
    encoded_revision = urllib.parse.quote(revision, safe="")
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{encoded_revision}"
    if path:
        encoded_path = urllib.parse.quote(path)
        url = f"{url}/{encoded_path}"
    return f"{url}?recursive=true"


def list_repo_files(repo_id: str, revision: str, token: str | None, path: str | None = None) -> list[str]:
    if HfApi is not None:
        api = HfApi()
        files = api.list_repo_files(
            repo_id=repo_id,
            revision=revision,
            repo_type=DEFAULT_REPO_TYPE,
            token=token,
        )
        if path:
            prefix = path.rstrip("/") + "/"
            return [file_path for file_path in files if file_path.startswith(prefix)]
        return files

    request = urllib.request.Request(hf_api_tree_url(repo_id, revision, path), headers=auth_headers(token))
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Hugging Face API response while listing repository files.")
    return [item["path"] for item in payload if isinstance(item, dict) and "path" in item]


def download_remote_file(
    repo_id: str,
    revision: str,
    token: str | None,
    remote_path: str,
    destination: Path,
) -> Path:
    if hf_hub_download is not None:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type=DEFAULT_REPO_TYPE,
                revision=revision,
                token=token,
                filename=remote_path,
                local_dir=destination.parent,
                local_dir_use_symlinks=False,
            )
        )

    request = urllib.request.Request(hf_url(repo_id, revision, remote_path), headers=auth_headers(token))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total else None
        downloaded = 0
        next_report = 0
        report_step = 256 * 1024 * 1024
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                next_report = downloaded + report_step
                if total_bytes:
                    percent = min(100.0, 100.0 * downloaded / total_bytes)
                    print(
                        f"      {downloaded / 1e9:.2f} GB / "
                        f"{total_bytes / 1e9:.2f} GB ({percent:.1f}%)"
                    )
                else:
                    print(f"      {downloaded / 1e9:.2f} GB")
    return destination


def download_scene(
    scene_id: str,
    remote_files: list[str],
    *,
    repo_id: str,
    revision: str,
    token: str | None,
    ncore_root: Path,
    force: bool,
    skip_existing: bool,
    dry_run: bool,
) -> bool:
    scene_dir = ncore_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and remote_files:
        targets = [
            scene_dir / Path(path).relative_to(scene_target_prefix(scene_id))
            for path in remote_files
        ]
        if all(path.exists() for path in targets):
            print(f"[+] skipping scene {scene_id} (requested files already exist)")
            return True

    print(f"==> scene {scene_id}")
    if dry_run:
        for path in remote_files:
            print(f"   would-download {path}")
        return True

    downloaded = 0
    with tempfile.TemporaryDirectory(prefix=f"ncore-{scene_id}-") as tmp:
        tmp_dir = Path(tmp)
        for remote_path in remote_files:
            target = scene_dir / Path(remote_path).relative_to(scene_target_prefix(scene_id))
            if target.exists() and not force:
                print(f"   skip existing file: {target}")
                continue

            local_path = download_remote_file(
                repo_id,
                revision,
                token,
                remote_path,
                tmp_dir / Path(remote_path).name,
            )
            if force and target.exists():
                target.unlink()
            copied = copy_downloaded_file(local_path, scene_dir, scene_id, remote_path)
            downloaded += 1
            print(f"   downloaded {copied.name}")

    if downloaded:
        return True

    required_targets = [scene_dir / Path(path).relative_to(scene_target_prefix(scene_id)) for path in remote_files]
    return all(path.exists() for path in required_targets)


def choose_scenes(args: argparse.Namespace, repo_scene_ids: list[str]) -> list[str]:
    if args.all:
        selected = list(repo_scene_ids)
        if args.max_scenes > 0:
            selected = selected[: args.max_scenes]
        return selected

    requested: list[str] = []
    if args.scene_id:
        requested.extend(parse_scene_tokens(args.scene_id))
    if args.scene_id_file:
        requested.extend(parse_scene_file(args.scene_id_file))
    requested = dedupe(requested)

    if not requested:
        raise SystemExit("No scene selected. Use --all, or pass --scene-id / --scene-id-file.")

    missing = [scene for scene in requested if scene not in repo_scene_ids]
    if missing:
        raise ValueError(f"Scene(s) not found in repo index: {', '.join(missing)}")

    if args.max_scenes > 0:
        requested = requested[: args.max_scenes]
    return requested


def main() -> None:
    args = parse_args()
    token = resolve_token(args.token)

    try:
        repo_files = list_repo_files(args.repo_id, args.revision, token)
    except (HfHubHTTPError, urllib.error.HTTPError) as exc:
        raise RuntimeError(
            "Unable to list repository files. Confirm HF token and dataset license acceptance."
        ) from exc

    repo_scene_ids = discover_scene_ids(repo_files)
    if not repo_scene_ids:
        raise RuntimeError("No scene ids were discovered in the repo index.")

    scenes = choose_scenes(args, repo_scene_ids)

    print("Repository   :", args.repo_id)
    print("Revision     :", args.revision)
    print("Mode         :", args.mode)
    print("Destination  :", args.ncore_root)
    print("Scenes       :", len(scenes))
    print("Token set    :", "yes" if token else "no")
    if args.dry_run:
        print("Dry-run mode : yes (no downloads will be performed)")

    output_root = args.ncore_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    done = 0
    for scene_id in scenes:
        try:
            scene_repo_files = list_repo_files(
                args.repo_id,
                args.revision,
                token,
                path=f"clips/{scene_id}",
            )
            remote_paths = files_for_scene(scene_id, scene_repo_files, args.mode)
            if download_scene(
                scene_id,
                remote_paths,
                repo_id=args.repo_id,
                revision=args.revision,
                token=token,
                ncore_root=output_root,
                force=args.force,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
            ):
                done += 1
            else:
                failed.append(f"{scene_id}: no files downloaded")
        except Exception as exc:
            failed.append(f"{scene_id}: {exc}")

    print("----")
    print(f"Succeeded : {done}")
    print(f"Failed    : {len(failed)}")
    if failed:
        for item in failed:
            print(f" - {item}")
        raise SystemExit(1)

    if scenes:
        print(f"Done. Scenes available in {output_root}")


if __name__ == "__main__":
    main()
