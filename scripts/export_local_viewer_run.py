#!/usr/bin/env python3

"""Export a VBOGS run bundle that can be opened by the local viewer."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVE = "2013_05_28_drive_0008_sync"
DEFAULT_COLMAP_ROOT = Path("/data/COLMAP")
DEFAULT_OCTREE_ROOT = Path("/data/OCTREE-ANYGS")
DEFAULT_BUCKET_ROOT = REPO_ROOT / "data" / "m4"
DEFAULT_EXPORT_ROOT = REPO_ROOT / "outputs" / "local_viewer_exports"
OPTIONAL_MODEL_ROOT_FILES = ("input.ply", "cameras.json", "cfg_args")
OPTIONAL_UNCERTAINTY_FILES = (
    "uncertainty_metadata.json",
    "uncertainty_components.npz",
    "uncertainty_histogram.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", default=DEFAULT_DRIVE)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Octree-AnyGS run directory. Defaults to latest run under /data/OCTREE-ANYGS/<drive>.",
    )
    parser.add_argument(
        "--octree-output-root",
        type=Path,
        default=DEFAULT_OCTREE_ROOT,
        help="Root containing Octree-AnyGS runs when --model-path is omitted.",
    )
    parser.add_argument(
        "--colmap-root",
        type=Path,
        default=DEFAULT_COLMAP_ROOT,
        help="Root containing prepared COLMAP scenes.",
    )
    parser.add_argument(
        "--colmap-path",
        type=Path,
        default=None,
        help="Prepared COLMAP scene directory. Defaults to <colmap-root>/<drive>.",
    )
    parser.add_argument(
        "--bucket-root",
        type=Path,
        default=None,
        help="Directory with M4/M5 artifacts. Defaults to data/m4/<drive>.",
    )
    parser.add_argument(
        "--u-path",
        type=Path,
        default=None,
        help="Per-anchor uncertainty array. Defaults to <bucket-root>/U.npy.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Checkpoint iteration to export. -1 selects the latest checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Export directory. Defaults to outputs/local_viewer_exports/<drive>.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Zip path. Defaults to <output-dir>-local-viewer.zip.",
    )
    parser.add_argument("--no-archive", action="store_true", help="Skip writing the zip archive.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing export directory/archive.",
    )
    return parser.parse_args()


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_model_path(drive: str, model_path: Path | None, octree_output_root: Path) -> Path:
    if model_path is not None:
        resolved = model_path.resolve()
        if not (resolved / "config.yaml").is_file():
            raise FileNotFoundError(f"Octree config not found: {resolved / 'config.yaml'}")
        return resolved

    root = octree_output_root / drive
    if not root.exists():
        raise FileNotFoundError(f"Octree output root not found: {root}")
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and (path / "config.yaml").is_file())
    if not candidates:
        raise FileNotFoundError(f"No Octree-AnyGS runs found under {root}")
    return candidates[-1].resolve()


def resolve_colmap_path(drive: str, colmap_root: Path, colmap_path: Path | None) -> Path:
    if colmap_path is not None:
        resolved = colmap_path.resolve()
    else:
        resolved = (colmap_root / drive).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Prepared COLMAP scene not found: {resolved}")
    return resolved


def resolve_bucket_root(drive: str, bucket_root: Path | None) -> Path:
    if bucket_root is not None:
        return bucket_root.resolve()
    return (DEFAULT_BUCKET_ROOT / drive).resolve()


def resolve_u_path(bucket_root: Path, u_path: Path | None) -> Path:
    if u_path is not None:
        resolved = u_path.resolve()
    else:
        resolved = (bucket_root / "U.npy").resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Uncertainty array not found: {resolved}")
    return resolved


def parse_iteration_name(path: Path) -> int | None:
    prefix = "iteration_"
    if not path.name.startswith(prefix):
        return None
    try:
        return int(path.name[len(prefix) :])
    except ValueError:
        return None


def find_iteration_dir(model_path: Path, iteration: int) -> tuple[int, Path]:
    point_cloud_root = model_path / "point_cloud"
    if not point_cloud_root.is_dir():
        raise FileNotFoundError(f"Point-cloud checkpoint root not found: {point_cloud_root}")

    if iteration >= 0:
        iteration_dir = point_cloud_root / f"iteration_{iteration}"
        if not (iteration_dir / "point_cloud_anchor.ply").is_file():
            raise FileNotFoundError(f"Anchor checkpoint not found: {iteration_dir / 'point_cloud_anchor.ply'}")
        return iteration, iteration_dir

    candidates: list[tuple[int, Path]] = []
    for child in point_cloud_root.iterdir():
        parsed = parse_iteration_name(child)
        if parsed is not None and (child / "point_cloud_anchor.ply").is_file():
            candidates.append((parsed, child))
    if not candidates:
        raise FileNotFoundError(f"No anchor checkpoints found under {point_cloud_root}")
    return max(candidates, key=lambda item: item[0])


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def is_implicit_model(cfg: dict[str, Any]) -> bool:
    model_params = cfg.get("model_params", {})
    if not isinstance(model_params, dict):
        return False
    model_config = model_params.get("model_config", {})
    if not isinstance(model_config, dict):
        return False
    kwargs = model_config.get("kwargs", {})
    if not isinstance(kwargs, dict):
        return False
    return str(kwargs.get("gs_attr", "")).startswith("implicit")


def appearance_dim(cfg: dict[str, Any]) -> int:
    kwargs = cfg.get("model_params", {}).get("model_config", {}).get("kwargs", {})
    if not isinstance(kwargs, dict):
        return 0
    try:
        return int(kwargs.get("appearance_dim", 0) or 0)
    except (TypeError, ValueError):
        return 0


def required_checkpoint_files(cfg: dict[str, Any]) -> tuple[str, ...]:
    required = ["point_cloud_anchor.ply"]
    if is_implicit_model(cfg):
        required.extend(["opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"])
        if appearance_dim(cfg) > 0:
            required.append("embedding_appearance.pt")
    return tuple(required)


def copy_file(source: Path, destination: Path, *, required: bool, copied: list[dict[str, str]]) -> None:
    source = source.resolve()
    if not source.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {source}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied.append({"source": str(source), "destination": str(destination.resolve())})


def copy_tree(
    source: Path,
    destination: Path,
    *,
    copied: list[dict[str, str]],
    dereference_symlinks: bool = False,
) -> None:
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Required directory not found: {source}")
    shutil.copytree(source, destination, symlinks=not dereference_symlinks, dirs_exist_ok=True)
    copied.append(
        {
            "source": str(source),
            "destination": str(destination.resolve()),
            "symlinks": "dereferenced" if dereference_symlinks else "preserved",
        }
    )


def read_colmap_image_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"COLMAP images.txt not found: {path}")

    names: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10:
            names.append(parts[9])
    if not names:
        raise ValueError(f"No image entries found in {path}")
    return names


def validate_materialized_colmap_images(prepared_path: Path) -> dict[str, Any]:
    image_names = read_colmap_image_names(prepared_path / "sparse" / "0" / "images.txt")
    image_root = prepared_path / "images"
    missing = [name for name in image_names if not (image_root / name).is_file()]
    symlinked = [name for name in image_names if (image_root / name).is_symlink()]

    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} missing)"
        raise FileNotFoundError(f"Prepared image files are missing under {image_root}: {preview}{suffix}")
    if symlinked:
        preview = ", ".join(symlinked[:5])
        suffix = "" if len(symlinked) <= 5 else f", ... ({len(symlinked)} symlinks)"
        raise ValueError(
            f"Local-viewer exports must contain real image files, not symlinks: {preview}{suffix}"
        )

    return {
        "image_count": len(image_names),
        "image_root": str(image_root.resolve()),
        "materialized": True,
    }


def default_output_dir(drive: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return repo_path(output_dir)
    return (DEFAULT_EXPORT_ROOT / drive).resolve()


def default_archive_path(output_dir: Path, archive_path: Path | None) -> Path:
    if archive_path is not None:
        return repo_path(archive_path)
    return output_dir.parent / f"{output_dir.name}-local-viewer.zip"


def viewer_commands(output_dir: Path, iteration: int) -> str:
    export_var = display_path(output_dir)
    return f"""# VBOGS local viewer commands

This folder is a portable local viewer export. Download the matching
`*-local-viewer.zip`, extract it on a machine with a checked-out VBOGS repo,
and point `EXPORT_DIR` at the extracted `local_viewer` folder.

```bash
cd /path/to/VBOGS
EXPORT_DIR={export_var}

# If you are already inside the extracted local_viewer folder, use:
# EXPORT_DIR="$PWD"

python scripts/view_octree_anygs.py \\
  --model-path "${{EXPORT_DIR}}/model" \\
  --u-path "${{EXPORT_DIR}}/uncertainty/U.npy" \\
  --iteration {iteration} \\
  --resolution 4
```

Open the browser viewer at:

```text
http://localhost:8070
```

Query the loaded scene and available cameras:

```bash
curl http://localhost:8070/api/metadata
curl http://localhost:8070/api/cameras
```

Query rendered uncertainty totals for a saved camera:

```bash
curl -X POST http://localhost:8070/api/rendered-anchors \\
  -H 'Content-Type: application/json' \\
  -d '{{"camera_id":"test:0","max_anchors":25}}'
```

For one-off PNG diagnostics:

```bash
cd /path/to/VBOGS
EXPORT_DIR={export_var}

python scripts/render_uncertainty_views.py \\
  --model-path "${{EXPORT_DIR}}/model" \\
  --uncertainty "${{EXPORT_DIR}}/uncertainty/U.npy" \\
  --iteration {iteration} \\
  --split both \\
  --output-dir "${{EXPORT_DIR}}/rendered_views"
```
"""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def archive_export_dir(output_dir: Path, archive_path: Path, *, overwrite: bool) -> Path:
    if archive_path.exists():
        if not overwrite:
            raise FileExistsError(f"Archive already exists: {archive_path}. Use --overwrite to replace it.")
        archive_path.unlink()

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir.parent).as_posix())
    return archive_path.resolve()


def export_local_viewer_run(
    *,
    drive: str,
    model_path: Path | None,
    octree_output_root: Path,
    colmap_root: Path,
    colmap_path: Path | None,
    bucket_root: Path | None,
    u_path: Path | None,
    iteration: int,
    output_dir: Path | None,
    archive_path: Path | None,
    write_archive: bool,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = default_output_dir(drive, output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Export directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)

    source_model_path = resolve_model_path(drive, model_path, octree_output_root)
    source_colmap_path = resolve_colmap_path(drive, colmap_root, colmap_path)
    source_bucket_root = resolve_bucket_root(drive, bucket_root)
    source_u_path = resolve_u_path(source_bucket_root, u_path)
    resolved_iteration, source_iteration_dir = find_iteration_dir(source_model_path, iteration)

    cfg = load_yaml(source_model_path / "config.yaml")
    for name in required_checkpoint_files(cfg):
        required_path = source_iteration_dir / name
        if not required_path.is_file():
            raise FileNotFoundError(f"Required checkpoint file not found: {required_path}")

    copied: list[dict[str, str]] = []
    model_dest = output_dir / "model"
    prepared_dest = output_dir / "prepared"
    uncertainty_dest = output_dir / "uncertainty"

    copy_tree(source_colmap_path, prepared_dest, copied=copied, dereference_symlinks=True)
    prepared_images = validate_materialized_colmap_images(prepared_dest)
    copy_file(source_model_path / "config.yaml", model_dest / "original_config.yaml", required=True, copied=copied)

    patched_cfg = dict(cfg)
    patched_model_params = dict(patched_cfg.get("model_params", {}))
    patched_model_params["source_path"] = "../prepared"
    patched_cfg["model_params"] = patched_model_params
    write_yaml(model_dest / "config.yaml", patched_cfg)
    copied.append({"source": str((source_model_path / "config.yaml").resolve()), "destination": str((model_dest / "config.yaml").resolve())})

    for name in OPTIONAL_MODEL_ROOT_FILES:
        copy_file(source_model_path / name, model_dest / name, required=False, copied=copied)

    copy_tree(source_iteration_dir, model_dest / "point_cloud" / f"iteration_{resolved_iteration}", copied=copied)
    copy_file(source_u_path, uncertainty_dest / "U.npy", required=True, copied=copied)
    for name in OPTIONAL_UNCERTAINTY_FILES:
        copy_file(source_bucket_root / name, uncertainty_dest / name, required=False, copied=copied)

    commands_path = output_dir / "VIEWER_COMMANDS.md"
    commands_path.write_text(viewer_commands(output_dir, resolved_iteration), encoding="utf-8")

    resolved_archive_path = default_archive_path(output_dir, archive_path) if write_archive else None
    manifest = {
        "format": "vbogs-local-viewer-export-v1",
        "drive": drive,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": resolved_iteration,
        "output_dir": str(output_dir.resolve()),
        "layout": {
            "model_path": "model",
            "prepared_path": "prepared",
            "uncertainty_path": "uncertainty/U.npy",
            "viewer_commands": "VIEWER_COMMANDS.md",
        },
        "source_paths": {
            "octree_model_path": str(source_model_path),
            "octree_iteration_dir": str(source_iteration_dir.resolve()),
            "prepared_colmap_path": str(source_colmap_path),
            "bucket_root": str(source_bucket_root),
            "u_path": str(source_u_path),
        },
        "local_viewer": {
            "model_path": display_path(model_dest),
            "u_path": display_path(uncertainty_dest / "U.npy"),
            "config_source_path": "../prepared",
        },
        "prepared_images": prepared_images,
        "copied_artifacts": copied,
    }
    if resolved_archive_path is not None:
        manifest["archive_path"] = str(resolved_archive_path.resolve())
    write_json(output_dir / "local_viewer_manifest.json", manifest)

    if resolved_archive_path is not None:
        archive_export_dir(output_dir, resolved_archive_path, overwrite=overwrite)
    return manifest


def main() -> None:
    args = parse_args()
    try:
        manifest = export_local_viewer_run(
            drive=args.drive,
            model_path=args.model_path,
            octree_output_root=args.octree_output_root,
            colmap_root=args.colmap_root,
            colmap_path=args.colmap_path,
            bucket_root=args.bucket_root,
            u_path=args.u_path,
            iteration=args.iteration,
            output_dir=args.output_dir,
            archive_path=args.archive_path,
            write_archive=not args.no_archive,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"Wrote {manifest['output_dir']}")
    if "archive_path" in manifest:
        print(f"Wrote {manifest['archive_path']}")
    print(f"Viewer command file: {Path(manifest['output_dir']) / 'VIEWER_COMMANDS.md'}")


if __name__ == "__main__":
    main()
