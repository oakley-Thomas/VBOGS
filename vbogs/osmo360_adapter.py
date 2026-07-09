"""Helpers for DJI Osmo 360 / COLMAP-derived VBOGS adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class PointCloud:
    xyz: np.ndarray
    rgb: np.ndarray


def read_ply_xyz_rgb(path: Path) -> PointCloud:
    """Read XYZ/RGB vertex properties from a PLY file."""

    try:
        from plyfile import PlyData
    except ModuleNotFoundError:
        return read_ascii_ply_xyz_rgb(path)

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    missing_xyz = {"x", "y", "z"} - names
    if missing_xyz:
        raise ValueError(f"{path} is missing PLY XYZ properties: {sorted(missing_xyz)}")

    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack(
            [
                np.asarray(vertex["red"], dtype=np.uint8),
                np.asarray(vertex["green"], dtype=np.uint8),
                np.asarray(vertex["blue"], dtype=np.uint8),
            ],
            axis=1,
        )
    else:
        rgb = np.full((xyz.shape[0], 3), 160, dtype=np.uint8)
    return PointCloud(xyz=xyz, rgb=rgb)


def read_colmap_points3d_txt(path: Path) -> PointCloud:
    """Read COLMAP text-format points3D.txt as XYZ/RGB arrays."""

    xyz_rows: list[tuple[float, float, float]] = []
    rgb_rows: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 8:
                raise ValueError(f"Malformed COLMAP points3D row in {path}: {stripped}")
            xyz_rows.append((float(fields[1]), float(fields[2]), float(fields[3])))
            rgb_rows.append((int(fields[4]), int(fields[5]), int(fields[6])))
    if not xyz_rows:
        raise RuntimeError(f"COLMAP sparse point file contains no points: {path}")
    return PointCloud(
        xyz=np.asarray(xyz_rows, dtype=np.float32),
        rgb=np.asarray(rgb_rows, dtype=np.uint8),
    )


def write_binary_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary PLY with XYZ, zero normals, and RGB properties."""

    try:
        from plyfile import PlyData, PlyElement
    except ModuleNotFoundError:
        write_ascii_ply(path, xyz, rgb)
        return

    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must have matching row counts")
    normals = np.zeros_like(xyz, dtype=np.float32)
    vertex_data = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex_data[:] = list(map(tuple, np.concatenate([xyz, normals, rgb], axis=1)))
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex_data, "vertex")], text=False).write(path)


def read_ascii_ply_xyz_rgb(path: Path) -> PointCloud:
    """Read the simple ASCII PLY subset used by local tests."""

    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        vertex_count = None
        properties: list[str] = []
        ascii_format = False
        for line in handle:
            stripped = line.strip()
            if stripped == "format ascii 1.0":
                ascii_format = True
            elif stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            elif stripped.startswith("property "):
                properties.append(stripped.split()[-1])
            elif stripped == "end_header":
                break
        if not ascii_format:
            raise ModuleNotFoundError(
                "plyfile is required to read binary PLY files; install plyfile or use ASCII PLY"
            )
        if vertex_count is None:
            raise ValueError(f"Missing vertex count in PLY header: {path}")
        rows = [handle.readline().strip().split() for _ in range(vertex_count)]

    prop_index = {name: index for index, name in enumerate(properties)}
    for name in ("x", "y", "z"):
        if name not in prop_index:
            raise ValueError(f"{path} is missing PLY property {name}")
    xyz = np.asarray(
        [
            [
                float(row[prop_index["x"]]),
                float(row[prop_index["y"]]),
                float(row[prop_index["z"]]),
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    if {"red", "green", "blue"}.issubset(prop_index):
        rgb = np.asarray(
            [
                [
                    int(row[prop_index["red"]]),
                    int(row[prop_index["green"]]),
                    int(row[prop_index["blue"]]),
                ]
                for row in rows
            ],
            dtype=np.uint8,
        )
    else:
        rgb = np.full((xyz.shape[0], 3), 160, dtype=np.uint8)
    return PointCloud(xyz=xyz, rgb=rgb)


def write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must have matching row counts")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {xyz.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for point, color in zip(xyz, rgb):
        lines.append(
            f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
            f"{int(color[0])} {int(color[1])} {int(color[2])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def list_images(path: Path) -> list[Path]:
    return sorted(child for child in path.rglob("*") if child.suffix.lower() in IMAGE_EXTENSIONS)


def count_colmap_images_txt(path: Path) -> int:
    """Count registered image rows in COLMAP images.txt."""

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # COLMAP images.txt alternates metadata and 2D-point rows.
            if line_index >= 0:
                fields = stripped.split()
                if len(fields) >= 10:
                    count += 1
    return count


def command_to_string(command: Iterable[object]) -> str:
    return " ".join(str(part) for part in command)
