#!/usr/bin/env python3

"""Download and extract KITTI/KITTI-360 archives into `data/KITTI-360`.

This script is intentionally manifest-driven so deployment-specific URLs can be
filled in without changing the code. It uses only the Python standard library
so it can run inside the repo containers without extra packages.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "KITTI-360"
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "download_manifest.json"
DEFAULT_EXAMPLE_MANIFEST = REPO_ROOT / "scripts" / "kitti360_download_manifest.example.json"
CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=(
            "JSON manifest describing which archives to download. "
            "Default: data/KITTI-360/download_manifest.json"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory where KITTI-360 data should live.",
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=None,
        help="Optional directory for cached archives. Defaults to <data-root>/_downloads.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download archives even if they are already present in the cache.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archives after extraction.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a manifest entry when all of its expected_paths already exist.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}\n"
            f"Copy {DEFAULT_EXAMPLE_MANIFEST} to {DEFAULT_MANIFEST} and fill in the URLs."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_list(value: Any, *, field_name: str) -> List[Any]:
    if not isinstance(value, list):
        raise ValueError(f"`{field_name}` must be a list")
    return value


def validate_entry(entry: Dict[str, Any]) -> None:
    required_fields = ("name", "url")
    for field in required_fields:
        if field not in entry:
            raise ValueError(f"Manifest entry missing required field `{field}`: {entry}")
    if "PASTE_" in str(entry["url"]) or "example.com" in str(entry["url"]):
        raise ValueError(
            f"Manifest entry `{entry['name']}` still contains a placeholder URL: {entry['url']}"
        )
    if "expected_paths" in entry:
        ensure_list(entry["expected_paths"], field_name="expected_paths")


def archive_name_for_entry(entry: Dict[str, Any]) -> str:
    explicit_name = entry.get("archive_name")
    if explicit_name:
        return str(explicit_name)
    parsed = urllib.parse.urlparse(str(entry["url"]))
    name = Path(parsed.path).name
    if not name:
        raise ValueError(f"Could not infer archive name from URL: {entry['url']}")
    return name


def print_progress(downloaded: int, total: int | None, *, prefix: str) -> None:
    if total and total > 0:
        percent = min(100.0, downloaded * 100.0 / total)
        message = f"\r{prefix}: {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({percent:5.1f}%)"
    else:
        message = f"\r{prefix}: {downloaded / 1e6:.1f} MB"
    sys.stdout.write(message)
    sys.stdout.flush()


def download_file(url: str, destination: Path, *, force_download: bool) -> None:
    if destination.exists() and not force_download:
        print(f"Using cached archive: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None
        downloaded = 0
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            print_progress(downloaded, total_bytes, prefix=f"  -> {destination.name}")
    print("")


def iter_extraction_members_tar(archive: tarfile.TarFile, strip_components: int) -> Iterable[tarfile.TarInfo]:
    for member in archive.getmembers():
        parts = Path(member.name).parts
        if len(parts) <= strip_components:
            continue
        trimmed_parts = parts[strip_components:]
        member = copy.copy(member)
        member.name = str(Path(*trimmed_parts))
        yield member


def iter_extraction_members_zip(archive: zipfile.ZipFile, strip_components: int) -> Iterable[zipfile.ZipInfo]:
    for member in archive.infolist():
        parts = Path(member.filename).parts
        if len(parts) <= strip_components:
            continue
        trimmed_parts = parts[strip_components:]
        member.filename = str(Path(*trimmed_parts))
        yield member


def extract_archive(archive_path: Path, destination: Path, *, strip_components: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    suffixes = archive_path.suffixes

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            for member in iter_extraction_members_zip(archive, strip_components):
                archive.extract(member, path=destination)
        return

    if suffixes[-2:] in ([".tar", ".gz"], [".tar", ".bz2"], [".tar", ".xz"]) or archive_path.suffix == ".tar":
        with tarfile.open(archive_path) as archive:
            for member in iter_extraction_members_tar(archive, strip_components):
                archive.extract(member, path=destination)
        return

    raise ValueError(f"Unsupported archive format: {archive_path}")


def expected_paths_exist(entry: Dict[str, Any], data_root: Path) -> bool:
    expected_paths = entry.get("expected_paths", [])
    if not expected_paths:
        return False
    return all((data_root / rel_path).exists() for rel_path in expected_paths)


def verify_expected_paths(entry: Dict[str, Any], data_root: Path) -> None:
    missing = [rel_path for rel_path in entry.get("expected_paths", []) if not (data_root / rel_path).exists()]
    if missing:
        raise FileNotFoundError(
            f"Entry `{entry['name']}` completed but expected paths are still missing: {missing}"
        )


def process_entry(
    entry: Dict[str, Any],
    *,
    data_root: Path,
    downloads_dir: Path,
    force_download: bool,
    keep_archives: bool,
    skip_existing: bool,
) -> None:
    validate_entry(entry)

    if skip_existing and expected_paths_exist(entry, data_root):
        print(f"Skipping `{entry['name']}` because expected paths already exist")
        return

    archive_name = archive_name_for_entry(entry)
    archive_path = downloads_dir / archive_name
    extract_to = data_root / entry.get("extract_to", ".")
    strip_components = int(entry.get("strip_components", 0))

    download_file(str(entry["url"]), archive_path, force_download=force_download)

    if entry.get("skip_extract", False):
        print(f"Skipping extraction for `{entry['name']}`")
    else:
        print(f"Extracting `{entry['name']}` into {extract_to}")
        extract_archive(archive_path, extract_to, strip_components=strip_components)

    verify_expected_paths(entry, data_root)

    if keep_archives or entry.get("keep_archive", False):
        return

    if archive_path.exists():
        archive_path.unlink()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest.resolve())
    downloads = ensure_list(manifest.get("downloads", []), field_name="downloads")

    data_root = args.data_root.resolve()
    downloads_dir = (args.downloads_dir or (data_root / "_downloads")).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root     : {data_root}")
    print(f"Manifest      : {args.manifest.resolve()}")
    print(f"Downloads dir : {downloads_dir}")

    for entry in downloads:
        print("")
        print(f"==> {entry['name']}")
        process_entry(
            entry,
            data_root=data_root,
            downloads_dir=downloads_dir,
            force_download=args.force_download,
            keep_archives=args.keep_archives,
            skip_existing=args.skip_existing,
        )

    print("")
    print("KITTI-360 download manifest completed successfully")


if __name__ == "__main__":
    main()
