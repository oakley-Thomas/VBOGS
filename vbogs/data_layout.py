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


def resolve_kitti360_path(explicit_path: Path | None, *, kind: str) -> Path:
    """Resolve a KITTI-360 input path from an explicit override or repo defaults."""

    if explicit_path is not None:
        return explicit_path

    for candidate in _candidate_paths(kind):
        if candidate.exists():
            return candidate

    return _candidate_paths(kind)[0]
