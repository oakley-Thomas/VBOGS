#!/usr/bin/env python3

"""Package compact exp02 review artifacts per drive."""

from __future__ import annotations

import argparse
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVE_GLOB = "2013_05_28_drive_*_sync"
VARIANTS = ("explicit", "implicit")
DEFAULT_EXPERIMENTS_ROOT = REPO_ROOT / "outputs" / "experiments"
DEFAULT_ROOTS = {
    "explicit": DEFAULT_EXPERIMENTS_ROOT / "exp02_explicit3d_baseline",
    "implicit": DEFAULT_EXPERIMENTS_ROOT / "exp02_implicit3d_baseline",
}
DEFAULT_CONFIGS = {
    "explicit": DEFAULT_EXPERIMENTS_ROOT / "exp02-A_explicit3d_baseline_pipeline_config.yaml",
    "implicit": DEFAULT_EXPERIMENTS_ROOT / "exp02_B_implicit3d_baseline_pipeline_config.yaml",
}
REQUIRED_RELATIVE_DIRS = (
    Path("views") / "train" / "side_by_side",
    Path("views") / "test" / "side_by_side",
    Path("pointclouds") / "anchors",
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    root: Path
    config: Path


def repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def selected_variant_names(raw_variant: str) -> list[str]:
    if raw_variant == "both":
        return list(VARIANTS)
    return [raw_variant]


def build_variant_specs(args: argparse.Namespace) -> list[VariantSpec]:
    roots = {
        "explicit": repo_path(args.explicit_root) if args.explicit_root else DEFAULT_ROOTS["explicit"],
        "implicit": repo_path(args.implicit_root) if args.implicit_root else DEFAULT_ROOTS["implicit"],
    }
    configs = {
        "explicit": repo_path(args.explicit_config) if args.explicit_config else DEFAULT_CONFIGS["explicit"],
        "implicit": repo_path(args.implicit_config) if args.implicit_config else DEFAULT_CONFIGS["implicit"],
    }
    return [
        VariantSpec(name=name, root=roots[name], config=configs[name])
        for name in selected_variant_names(args.variant)
    ]


def dedupe_preserving_order(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def discover_drives(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment root not found: {root}")
    return sorted(path.name for path in root.glob(DRIVE_GLOB) if path.is_dir())


def drive_names_for_spec(spec: VariantSpec, requested_drives: Sequence[str] | None) -> list[str]:
    if requested_drives:
        return dedupe_preserving_order(requested_drives)
    drives = discover_drives(spec.root)
    if not drives:
        raise RuntimeError(f"No drive directories matching {DRIVE_GLOB} found under {spec.root}")
    return drives


def output_zip_path(spec: VariantSpec, drive: str) -> Path:
    return spec.root / f"{drive}-{spec.name}.zip"


def validate_package_inputs(spec: VariantSpec, drive: str) -> Path:
    drive_root = spec.root / drive
    if not drive_root.is_dir():
        raise FileNotFoundError(f"Drive output directory not found: {drive_root}")
    if not spec.config.is_file():
        raise FileNotFoundError(f"Pipeline config not found: {spec.config}")
    for relative_dir in REQUIRED_RELATIVE_DIRS:
        path = drive_root / relative_dir
        if not path.is_dir():
            raise FileNotFoundError(f"Required artifact directory not found: {path}")
    return drive_root


def iter_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def package_drive_artifacts(
    spec: VariantSpec,
    drive: str,
    *,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, object]:
    drive_root = validate_package_inputs(spec, drive)
    archive_path = output_zip_path(spec, drive)
    top_level = f"{drive}-{spec.name}"

    if archive_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output already exists: {archive_path}. Use --overwrite to replace it.")

    archive_members: list[str] = [f"{top_level}/pipeline_config.yaml"]
    for relative_dir in REQUIRED_RELATIVE_DIRS:
        source_dir = drive_root / relative_dir
        for source_path in iter_files(source_dir):
            archive_members.append(f"{top_level}/{source_path.relative_to(drive_root).as_posix()}")

    if dry_run:
        return {
            "drive": drive,
            "variant": spec.name,
            "archive_path": str(archive_path),
            "member_count": len(archive_members),
            "dry_run": True,
        }

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(spec.config, f"{top_level}/pipeline_config.yaml")
        for relative_dir in REQUIRED_RELATIVE_DIRS:
            source_dir = drive_root / relative_dir
            for source_path in iter_files(source_dir):
                archive.write(source_path, f"{top_level}/{source_path.relative_to(drive_root).as_posix()}")

    return {
        "drive": drive,
        "variant": spec.name,
        "archive_path": str(archive_path.resolve()),
        "member_count": len(archive_members),
        "dry_run": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("explicit", "implicit", "both"),
        default="both",
        help="Exp02 variant to package.",
    )
    parser.add_argument(
        "--drive",
        action="append",
        dest="drives",
        default=None,
        help="Drive id to package. May be repeated. Defaults to all drives in selected roots.",
    )
    parser.add_argument("--explicit-root", type=Path, default=None)
    parser.add_argument("--implicit-root", type=Path, default=None)
    parser.add_argument("--explicit-config", type=Path, default=None)
    parser.add_argument("--implicit-config", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing zip files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned archives without writing.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue packaging later drives after a failure.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    failures: list[tuple[str, str, Exception]] = []

    for spec in build_variant_specs(args):
        drives = drive_names_for_spec(spec, args.drives)
        for drive in drives:
            try:
                summary = package_drive_artifacts(
                    spec,
                    drive,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )
            except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
                if not args.continue_on_error:
                    raise SystemExit(f"error: {exc}") from exc
                failures.append((spec.name, drive, exc))
                print(f"error: {drive} ({spec.name}): {exc}", file=sys.stderr, flush=True)
                continue

            action = "Would write" if args.dry_run else "Wrote"
            print(
                f"{action} {summary['archive_path']} "
                f"with {summary['member_count']} members",
                flush=True,
            )

    if failures:
        print("\nFailed exp02 packages:", file=sys.stderr)
        for variant, drive, exc in failures:
            print(f"  - {drive} ({variant}): {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
