#!/usr/bin/env python3

"""Copy curated pipeline artifacts into a versioned run output directory."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_local_viewer_run import export_local_viewer_run

DEFAULT_RUN_ROOT = REPO_ROOT / "outputs" / "v1_0"
DEFAULT_POINTS_ROOT = REPO_ROOT / "data" / "points_world"
DEFAULT_BUCKET_ROOT = REPO_ROOT / "data" / "m4"
DEFAULT_COLMAP_ROOT = Path("/data/COLMAP")
DEFAULT_OCTREE_ROOT = Path("/data/OCTREE-ANYGS")
ARCHIVE_EXCLUDED_TOP_LEVEL_DIRS = frozenset({"views"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive",
        default="2013_05_28_drive_0007_sync",
        help="KITTI-360 drive id to bundle.",
    )
    parser.add_argument(
        "--run-output-dir",
        type=Path,
        default=None,
        help="Final run output directory. Defaults to `outputs/v1_0/<drive>`.",
    )
    parser.add_argument(
        "--points-root",
        type=Path,
        default=DEFAULT_POINTS_ROOT,
        help="Root containing `data/points_world/<drive>` artifacts.",
    )
    parser.add_argument(
        "--bucket-root",
        type=Path,
        default=None,
        help="Directory containing M4/M5 artifacts. Defaults to `data/m4/<drive>`.",
    )
    parser.add_argument(
        "--colmap-root",
        type=Path,
        default=DEFAULT_COLMAP_ROOT,
        help="Root containing prepared COLMAP datasets.",
    )
    parser.add_argument(
        "--octree-output-root",
        type=Path,
        default=DEFAULT_OCTREE_ROOT,
        help="Root containing Octree-AnyGS training outputs.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Explicit Octree-AnyGS run directory. Defaults to latest run for the drive.",
    )
    parser.add_argument(
        "--map-viz-output-dir",
        type=Path,
        default=None,
        help="Directory containing anchor uncertainty PLY outputs.",
    )
    parser.add_argument(
        "--render-output-dir",
        type=Path,
        default=None,
        help="Directory containing rendered RGB/uncertainty/side-by-side views.",
    )
    parser.add_argument(
        "--nbv-output-dir",
        type=Path,
        default=None,
        help="Directory containing NBV scores and diagnostics.",
    )
    parser.add_argument(
        "--viewer-export-iteration",
        type=int,
        default=-1,
        help="Checkpoint iteration to include in the local viewer export. -1 selects latest.",
    )
    parser.add_argument(
        "--viewer-export-output-dir",
        type=Path,
        default=None,
        help="Local viewer export directory. Defaults to `<run-output-dir>/local_viewer`.",
    )
    parser.add_argument(
        "--viewer-export-archive-path",
        type=Path,
        default=None,
        help=(
            "Deprecated for bundle mode. The local viewer export is now included "
            "inside the single run zip."
        ),
    )
    parser.add_argument(
        "--skip-local-viewer-export",
        action="store_true",
        help="Skip creating the portable local viewer export during bundle.",
    )
    return parser.parse_args()


def resolve_run_output_dir(drive: str, run_output_dir: Path | None) -> Path:
    if run_output_dir is not None:
        return run_output_dir.resolve()
    return (DEFAULT_RUN_ROOT / drive).resolve()


def resolve_bucket_root(drive: str, bucket_root: Path | None) -> Path:
    if bucket_root is not None:
        return bucket_root.resolve()
    return (DEFAULT_BUCKET_ROOT / drive).resolve()


def resolve_drive_dir(root: Path, drive: str) -> Path:
    return (root / drive).resolve()


def resolve_colmap_drive_dir(colmap_root: Path, drive: str) -> Path:
    primary = colmap_root / drive
    if primary.exists():
        return primary.resolve()
    repo_fallback = REPO_ROOT / "data" / "COLMAP" / drive
    if repo_fallback.exists():
        return repo_fallback.resolve()
    return primary.resolve()


def resolve_octree_model_path(
    drive: str,
    *,
    model_path: Path | None,
    octree_output_root: Path,
) -> Path:
    if model_path is not None:
        return model_path.resolve()

    roots = [octree_output_root / drive, REPO_ROOT / "data" / "OCTREE-ANYGS" / drive]
    for root in roots:
        if not root.exists():
            continue
        candidates = sorted(
            path for path in root.iterdir() if path.is_dir() and (path / "config.yaml").exists()
        )
        if candidates:
            return candidates[-1].resolve()

    return (octree_output_root / drive / "<latest>").resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def copy_file(
    source: Path,
    destination: Path,
    *,
    required: bool,
    copied: list[dict[str, str]],
    missing_optional: list[str],
) -> None:
    source = source.resolve()
    if not source.exists():
        if required:
            raise FileNotFoundError(f"Required artifact not found: {source}")
        missing_optional.append(str(source))
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied.append({"source": str(source), "destination": str(destination.resolve())})


def summarize_prepared(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    metadata = read_json(metadata_path)
    selected_frames = metadata.get("selected_frames")
    return {
        "num_frames": int(metadata.get("num_frames", 0) or 0),
        "selected_frame_count": len(selected_frames) if isinstance(selected_frames, list) else 0,
        "first_frame": selected_frames[0] if isinstance(selected_frames, list) and selected_frames else None,
        "last_frame": selected_frames[-1] if isinstance(selected_frames, list) and selected_frames else None,
    }


def summarize_stereo(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    metadata = read_json(metadata_path)
    return {
        "dataset": metadata.get("dataset"),
        "point_source": metadata.get("point_source") or "stereo",
        "num_frames": int(metadata.get("num_frames", 0) or 0),
        "num_points": int(metadata.get("num_points", 0) or 0),
        "matcher": metadata.get("matcher"),
        "output_npz": metadata.get("output_npz"),
        "output_ply": metadata.get("output_ply"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def default_archive_path(run_output_dir: Path) -> Path:
    run_output_dir = run_output_dir.resolve()
    return run_output_dir / f"{run_output_dir.name}.zip"


def default_viewer_export_output_dir(run_output_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    return (run_output_dir / "local_viewer").resolve()


def archive_run_output_dir(run_output_dir: Path) -> Path:
    run_output_dir = run_output_dir.resolve()
    if not run_output_dir.is_dir():
        raise NotADirectoryError(f"Run output directory not found: {run_output_dir}")

    archive_path = default_archive_path(run_output_dir)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_output_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.resolve() == archive_path:
                continue
            relative_to_run = path.relative_to(run_output_dir)
            if (
                relative_to_run.parts
                and relative_to_run.parts[0] in ARCHIVE_EXCLUDED_TOP_LEVEL_DIRS
            ):
                continue
            archive_name = Path(run_output_dir.name) / relative_to_run
            archive.write(path, archive_name.as_posix())
    return archive_path.resolve()


def bundle_run_outputs(
    *,
    drive: str,
    run_output_dir: Path,
    points_root: Path,
    bucket_root: Path,
    colmap_root: Path,
    octree_output_root: Path,
    model_path: Path | None,
    map_viz_output_dir: Path | None,
    render_output_dir: Path | None,
    nbv_output_dir: Path | None,
    viewer_export_iteration: int,
    viewer_export_output_dir: Path | None,
    viewer_export_archive_path: Path | None,
    skip_local_viewer_export: bool,
) -> dict[str, Any]:
    run_output_dir = run_output_dir.resolve()
    points_dir = resolve_drive_dir(points_root, drive)
    colmap_drive_dir = resolve_colmap_drive_dir(colmap_root, drive)
    resolved_model_path = resolve_octree_model_path(
        drive,
        model_path=model_path,
        octree_output_root=octree_output_root,
    )

    map_viz_dir = (
        map_viz_output_dir.resolve()
        if map_viz_output_dir is not None
        else (run_output_dir / "pointclouds" / "anchors").resolve()
    )
    render_dir = (
        render_output_dir.resolve()
        if render_output_dir is not None
        else (run_output_dir / "views").resolve()
    )
    nbv_dir = (
        nbv_output_dir.resolve()
        if nbv_output_dir is not None
        else (run_output_dir / "nbv").resolve()
    )

    copied: list[dict[str, str]] = []
    missing_optional: list[str] = []

    stereo_dest = run_output_dir / "pointclouds" / "stereo"
    copy_file(
        points_dir / "points_world.npz",
        stereo_dest / "points_world.npz",
        required=True,
        copied=copied,
        missing_optional=missing_optional,
    )
    copy_file(
        points_dir / "points_world.ply",
        stereo_dest / "points_world.ply",
        required=False,
        copied=copied,
        missing_optional=missing_optional,
    )
    copy_file(
        points_dir / "points_world_metadata.json",
        stereo_dest / "points_world_metadata.json",
        required=True,
        copied=copied,
        missing_optional=missing_optional,
    )

    uncertainty_dest = run_output_dir / "uncertainty"
    for name, required in (
        ("U.npy", True),
        ("uncertainty_components.npz", True),
        ("uncertainty_metadata.json", True),
        ("uncertainty_histogram.png", False),
    ):
        copy_file(
            bucket_root / name,
            uncertainty_dest / name,
            required=required,
            copied=copied,
            missing_optional=missing_optional,
        )

    prepared_metadata = colmap_drive_dir / "metadata.json"
    copy_file(
        prepared_metadata,
        run_output_dir / "prepared" / "metadata.json",
        required=True,
        copied=copied,
        missing_optional=missing_optional,
    )
    copy_file(
        resolved_model_path / "config.yaml",
        run_output_dir / "octree" / "config.yaml",
        required=True,
        copied=copied,
        missing_optional=missing_optional,
    )
    for name in ("results.json", "per_view.json"):
        copy_file(
            resolved_model_path / name,
            run_output_dir / "octree" / name,
            required=False,
            copied=copied,
            missing_optional=missing_optional,
        )

    local_viewer_export: dict[str, Any] | None = None
    if not skip_local_viewer_export:
        viewer_output_dir = default_viewer_export_output_dir(run_output_dir, viewer_export_output_dir)
        viewer_manifest = export_local_viewer_run(
            drive=drive,
            model_path=resolved_model_path,
            octree_output_root=octree_output_root,
            colmap_root=colmap_root,
            colmap_path=colmap_drive_dir,
            bucket_root=bucket_root,
            u_path=bucket_root / "U.npy",
            iteration=viewer_export_iteration,
            output_dir=viewer_output_dir,
            archive_path=viewer_export_archive_path,
            write_archive=False,
            overwrite=True,
        )
        local_viewer_export = {
            "output_dir": viewer_manifest["output_dir"],
            "viewer_commands": str((viewer_output_dir / "VIEWER_COMMANDS.md").resolve()),
            "manifest": str((viewer_output_dir / "local_viewer_manifest.json").resolve()),
            "iteration": viewer_manifest["iteration"],
            "model_path": str((viewer_output_dir / "model").resolve()),
            "uncertainty_path": str((viewer_output_dir / "uncertainty" / "U.npy").resolve()),
            "source_paths": viewer_manifest["source_paths"],
        }

    bundle_note = (
        "Bulky Octree-AnyGS checkpoints and full VBGS posterior files remain "
        "in their native data paths; source paths are recorded above."
    )
    if local_viewer_export is not None:
        bundle_note += (
            " The portable local viewer export is included under local_viewer/ "
            "inside the run zip, so the primary archive is renderable on a "
            "development machine."
        )
    else:
        bundle_note += (
            " The portable local viewer export was skipped, so the run zip is "
            "diagnostics-only and is not sufficient for local rendering."
        )

    points_summary = summarize_stereo(points_dir / "points_world_metadata.json")
    manifest = {
        "drive": drive,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_output_dir": str(run_output_dir),
        "archive": {
            "format": "zip",
            "path": str(default_archive_path(run_output_dir)),
            "base_dir": run_output_dir.name,
            "excluded_top_level_dirs": sorted(ARCHIVE_EXCLUDED_TOP_LEVEL_DIRS),
        },
        "frame_counts": summarize_prepared(prepared_metadata),
        "stereo": points_summary,
        "points": points_summary,
        "source_paths": {
            "points_dir": str(points_dir),
            "bucket_root": str(bucket_root.resolve()),
            "prepared_metadata": str(prepared_metadata.resolve()),
            "octree_model_path": str(resolved_model_path),
        },
        "stage_outputs": {
            "anchor_pointclouds": str(map_viz_dir),
            "rendered_views": str(render_dir),
            "nbv": str(nbv_dir),
        },
        "copied_artifacts": copied,
        "missing_optional_artifacts": missing_optional,
        "curated_bundle_note": bundle_note,
    }
    if local_viewer_export is not None:
        manifest["local_viewer_export"] = local_viewer_export
    write_json(run_output_dir / "run_manifest.json", manifest)
    archive_path = archive_run_output_dir(run_output_dir)
    manifest["archive"]["path"] = str(archive_path)
    write_json(run_output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    run_output_dir = resolve_run_output_dir(args.drive, args.run_output_dir)
    bucket_root = resolve_bucket_root(args.drive, args.bucket_root)
    manifest = bundle_run_outputs(
        drive=args.drive,
        run_output_dir=run_output_dir,
        points_root=args.points_root,
        bucket_root=bucket_root,
        colmap_root=args.colmap_root,
        octree_output_root=args.octree_output_root,
        model_path=args.model_path,
        map_viz_output_dir=args.map_viz_output_dir,
        render_output_dir=args.render_output_dir,
        nbv_output_dir=args.nbv_output_dir,
        viewer_export_iteration=args.viewer_export_iteration,
        viewer_export_output_dir=args.viewer_export_output_dir,
        viewer_export_archive_path=args.viewer_export_archive_path,
        skip_local_viewer_export=args.skip_local_viewer_export,
    )
    print(f"Wrote {run_output_dir / 'run_manifest.json'}")
    print(f"Wrote {manifest['archive']['path']}")
    print(
        "Bundle summary: "
        f"{len(manifest['copied_artifacts'])} copied, "
        f"{len(manifest['missing_optional_artifacts'])} optional missing"
    )


if __name__ == "__main__":
    main()
