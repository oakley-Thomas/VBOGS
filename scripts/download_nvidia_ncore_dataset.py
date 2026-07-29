#!/usr/bin/env python3
"""Download NVIDIA PhysicalAI AV NCore clips from Hugging Face.

The CLI remains useful for operators, while the implementation lives in
``vbogs.ncore_download`` so the authenticated web console can reuse it without
passing its backend token through a command line.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from vbogs.ncore_download import (
    DEFAULT_NCORE_ROOT,
    DEFAULT_REPO_ID,
    NCoreDownloadError,
    discover_scene_ids,
    download_scene,
    files_for_scene,
    list_repo_files,
)

DEFAULT_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGING_FACE_TOKEN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--ncore-root", type=Path, default=DEFAULT_NCORE_ROOT)
    parser.add_argument("--token", default=None, help="HF token (or use HF_TOKEN).")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--scene-id", action="append")
    parser.add_argument("--scene-id-file", type=Path)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--mode", choices=("core-only", "full"), default="core-only")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_scene_tokens(values: Iterable[str]) -> list[str]:
    return [token.strip() for value in values for token in str(value).split(",") if token.strip()]


def parse_scene_file(path: Path) -> list[str]:
    return parse_scene_tokens(
        raw for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip() and not raw.lstrip().startswith("#")
    )


def resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    return next((os.environ[name] for name in DEFAULT_TOKEN_ENV_VARS if os.environ.get(name)), None)


def dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def choose_scenes(args: argparse.Namespace, repo_scene_ids: list[str]) -> list[str]:
    if args.all:
        return repo_scene_ids[: args.max_scenes] if args.max_scenes > 0 else list(repo_scene_ids)
    requested = dedupe([
        *(parse_scene_tokens(args.scene_id) if args.scene_id else []),
        *(parse_scene_file(args.scene_id_file) if args.scene_id_file else []),
    ])
    if not requested:
        raise SystemExit("No scene selected. Use --all, or pass --scene-id / --scene-id-file.")
    missing = [scene for scene in requested if scene not in repo_scene_ids]
    if missing:
        raise ValueError(f"Scene(s) not found in repo index: {', '.join(missing)}")
    return requested[: args.max_scenes] if args.max_scenes > 0 else requested


def main() -> None:
    args = parse_args()
    token = resolve_token(args.token)
    if not token:
        raise SystemExit("An HF token is required. Set HF_TOKEN or pass --token.")
    repo_files = list_repo_files(args.repo_id, args.revision, token)
    scenes = choose_scenes(args, discover_scene_ids(repo_files))
    if not scenes:
        raise SystemExit("No scene ids were discovered in the repository index.")

    print("Repository   :", args.repo_id)
    print("Revision     :", args.revision)
    print("Mode         :", args.mode)
    print("Destination  :", args.ncore_root)
    print("Scenes       :", len(scenes))
    failed: list[str] = []
    for scene_id in scenes:
        try:
            remote = list_repo_files(args.repo_id, args.revision, token, path=f"clips/{scene_id}")
            ok = download_scene(
                scene_id, files_for_scene(scene_id, remote, args.mode), repo_id=args.repo_id,
                revision=args.revision, token=token, ncore_root=args.ncore_root, force=args.force,
                skip_existing=args.skip_existing, dry_run=args.dry_run, progress=print,
            )
            if not ok:
                failed.append(f"{scene_id}: no files downloaded")
        except Exception as exc:
            failed.append(f"{scene_id}: {exc}")
    print("Succeeded :", len(scenes) - len(failed))
    print("Failed    :", len(failed))
    if failed:
        print("\n".join(f" - {item}" for item in failed))
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except NCoreDownloadError as exc:
        raise SystemExit(str(exc)) from exc
