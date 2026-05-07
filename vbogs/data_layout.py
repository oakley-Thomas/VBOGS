"""Helpers for resolving repo-local dataset layouts.

These helpers keep the command-line entry points tolerant to small filesystem
reorganizations, especially around the KITTI-360 source tree.
"""

from __future__ import annotations

from pathlib import Path


def _candidate_paths(kind: str) -> list[Path]:
    if kind == "raw":
        return [
            Path("data/KITTI-360/data_2d_test"),
            Path("data/KITTI-360/images"),
            Path("data/data_2d_test"),
        ]
    if kind == "poses":
        return [
            Path("data/KITTI-360/data_poses"),
            Path("data/data_poses"),
        ]
    if kind == "calibration":
        return [
            Path("data/KITTI-360/calibration"),
            Path("data/calibration/calibration"),
            Path("data/calibration"),
        ]
    raise ValueError(f"Unknown layout kind: {kind}")


def _contains_expected_layout(path: Path, *, kind: str, drive: str | None) -> bool:
    if kind == "raw":
        if drive is None:
            return path.exists()
        return (
            (path / drive / "image_00" / "data_rect").exists()
            and (path / drive / "image_01" / "data_rect").exists()
        )
    if kind == "poses":
        if drive is None:
            return path.exists()
        return (path / drive / "cam0_to_world.txt").exists()
    if kind == "calibration":
        return (path / "perspective.txt").exists()
    raise ValueError(f"Unknown layout kind: {kind}")


def resolve_kitti360_path(
    explicit_path: Path | None,
    *,
    kind: str,
    drive: str | None = None,
) -> Path:
    """Resolve a KITTI-360 input path from an explicit override or repo defaults."""

    if explicit_path is not None:
        return explicit_path

    for candidate in _candidate_paths(kind):
        if _contains_expected_layout(candidate, kind=kind, drive=drive):
            return candidate

    return _candidate_paths(kind)[0]
