#!/usr/bin/env python3

"""Export map-scale per-anchor uncertainty as CloudCompare-friendly PLY files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.io import save_json
from vbogs.data_layout import resolve_kitti360_path

DEFAULT_DRIVE = "2013_05_28_drive_0008_sync"
PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("uncertainty", "<f4"),
        ("anchor_id", "<i4"),
        ("level", "<i2"),
        ("point_count", "<i4"),
        ("is_observed", "u1"),
        ("cell_size", "<f4"),
    ]
)
TRAJECTORY_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("frame_id", "<i4"),
        ("trajectory_index", "<i4"),
    ]
)
TRAJECTORY_EDGE_DTYPE = np.dtype([("vertex1", "<i4"), ("vertex2", "<i4")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive",
        default=DEFAULT_DRIVE,
        help="Drive id used to resolve default input/output paths.",
    )
    parser.add_argument(
        "--bucket-root",
        type=Path,
        default=None,
        help="Directory containing M4/M5 artifacts. Defaults to `data/m4/<drive>`.",
    )
    parser.add_argument(
        "--uncertainty",
        type=Path,
        default=None,
        help="Per-anchor uncertainty array. Defaults to `<bucket-root>/U.npy`.",
    )
    parser.add_argument(
        "--posterior",
        type=Path,
        default=None,
        help=(
            "Posterior artifact used for the observed-anchor mask. Defaults to "
            "`anchor_posterior.npz`, then `anchor_posterior.smoke.npz`."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to `outputs/uncertainty_maps/<drive>`.",
    )
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument(
        "--percentile-low",
        type=float,
        default=2.0,
        help="Observed-anchor percentile used for automatic color-scale minimum.",
    )
    parser.add_argument(
        "--percentile-high",
        type=float,
        default=98.0,
        help="Observed-anchor percentile used for automatic color-scale maximum.",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--include-unobserved",
        dest="observed_only",
        action="store_false",
        help="Include unobserved anchors and paint them red. This is the default.",
    )
    visibility.add_argument(
        "--observed-only",
        dest="observed_only",
        action="store_true",
        help="Export only anchors marked observed by the posterior artifact.",
    )
    parser.set_defaults(observed_only=False)
    parser.add_argument(
        "--no-split-levels",
        action="store_true",
        help="Only write the combined all-levels PLY.",
    )
    parser.add_argument(
        "--selection-metadata",
        type=Path,
        default=None,
        help=(
            "Prepared COLMAP metadata with `selected_frames`. Defaults to "
            "`/data/COLMAP/<drive>/metadata.json`."
        ),
    )
    parser.add_argument(
        "--poses-root",
        type=Path,
        default=None,
        help="Root containing KITTI-360 pose text files. Defaults to auto-detecting the repo layout.",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Skip writing `camera_trajectory.ply`.",
    )
    return parser.parse_args()


def resolve_bucket_root(drive: str, bucket_root: Path | None) -> Path:
    if bucket_root is not None:
        return bucket_root.resolve()
    return (REPO_ROOT / "data" / "m4" / drive).resolve()


def resolve_uncertainty_path(bucket_root: Path, uncertainty: Path | None) -> Path:
    if uncertainty is not None:
        return uncertainty.resolve()
    return (bucket_root / "U.npy").resolve()


def resolve_posterior_path(bucket_root: Path, posterior: Path | None) -> Path:
    if posterior is not None:
        return posterior.resolve()
    full = bucket_root / "anchor_posterior.npz"
    if full.exists():
        return full.resolve()
    smoke = bucket_root / "anchor_posterior.smoke.npz"
    if smoke.exists():
        return smoke.resolve()
    raise FileNotFoundError(
        f"Could not find `anchor_posterior.npz` or `anchor_posterior.smoke.npz` under {bucket_root}"
    )


def resolve_output_dir(drive: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    return (REPO_ROOT / "outputs" / "uncertainty_maps" / drive).resolve()


def resolve_selection_metadata(drive: str, selection_metadata: Path | None) -> Path:
    if selection_metadata is not None:
        return selection_metadata.resolve()
    return Path("/data/COLMAP") / drive / "metadata.json"


def _scalar_item(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"`{name}` must be scalar, got shape {array.shape}")
    return float(array.reshape(()))


def validate_color_scale_args(
    *,
    vmin: float | None,
    vmax: float | None,
    percentile_low: float,
    percentile_high: float,
) -> None:
    if (vmin is None) != (vmax is None):
        raise ValueError("Pass both --vmin and --vmax, or neither")
    if vmin is not None and vmax is not None and not vmin < vmax:
        raise ValueError(f"Expected --vmin < --vmax, got {vmin} >= {vmax}")
    if not 0.0 <= percentile_low <= 100.0:
        raise ValueError(f"--percentile-low must be in [0, 100], got {percentile_low}")
    if not 0.0 <= percentile_high <= 100.0:
        raise ValueError(f"--percentile-high must be in [0, 100], got {percentile_high}")
    if not percentile_low < percentile_high:
        raise ValueError(
            f"Expected --percentile-low < --percentile-high, got {percentile_low} >= {percentile_high}"
        )


def choose_color_scale(
    uncertainty: np.ndarray,
    is_observed: np.ndarray,
    *,
    vmin: float | None,
    vmax: float | None,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> tuple[float, float, str]:
    validate_color_scale_args(
        vmin=vmin,
        vmax=vmax,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax), "explicit"

    observed_finite = uncertainty[is_observed & np.isfinite(uncertainty)]
    scale_values = observed_finite
    source = "observed_percentiles"
    if scale_values.size == 0:
        scale_values = uncertainty[np.isfinite(uncertainty)]
        source = "all_finite_percentiles"
    if scale_values.size == 0:
        return 0.0, 1.0, "fallback_empty"

    lo = float(np.percentile(scale_values, percentile_low))
    hi = float(np.percentile(scale_values, percentile_high))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0, "fallback_nonfinite"
    if hi <= lo:
        pad = max(abs(lo) * 0.01, 1.0e-6)
        lo -= pad
        hi += pad
    return lo, hi, source


def uncertainty_to_rgb(
    uncertainty: np.ndarray,
    is_observed: np.ndarray,
    *,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    if not vmin < vmax:
        raise ValueError(f"Expected vmin < vmax, got {vmin} >= {vmax}")

    t = (np.asarray(uncertainty, dtype=np.float32) - np.float32(vmin)) / np.float32(vmax - vmin)
    t = np.clip(t, 0.0, 1.0)
    t = np.where(np.isfinite(t), t, 1.0)

    rgb = np.zeros((t.shape[0], 3), dtype=np.uint8)
    rgb[:, 0] = np.rint(t * 255.0).astype(np.uint8)
    rgb[:, 2] = np.rint((1.0 - t) * 255.0).astype(np.uint8)
    rgb[~is_observed] = np.array([255, 0, 0], dtype=np.uint8)
    return rgb


def build_vertex_array(
    *,
    anchor_xyz: np.ndarray,
    anchor_level: np.ndarray,
    point_counts: np.ndarray,
    uncertainty: np.ndarray,
    is_observed: np.ndarray,
    cell_size: np.ndarray,
    rgb: np.ndarray,
    anchor_ids: np.ndarray | None = None,
) -> np.ndarray:
    anchor_count = int(anchor_xyz.shape[0])
    if anchor_ids is None:
        anchor_ids = np.arange(anchor_count, dtype=np.int32)

    rows = np.empty(anchor_count, dtype=PLY_DTYPE)
    rows["x"] = anchor_xyz[:, 0].astype(np.float32)
    rows["y"] = anchor_xyz[:, 1].astype(np.float32)
    rows["z"] = anchor_xyz[:, 2].astype(np.float32)
    rows["red"] = rgb[:, 0]
    rows["green"] = rgb[:, 1]
    rows["blue"] = rgb[:, 2]
    rows["uncertainty"] = uncertainty.astype(np.float32)
    rows["anchor_id"] = anchor_ids.astype(np.int32)
    rows["level"] = anchor_level.astype(np.int16)
    rows["point_count"] = point_counts.astype(np.int32)
    rows["is_observed"] = is_observed.astype(np.uint8)
    rows["cell_size"] = cell_size.astype(np.float32)
    return rows


def ply_header(vertex_count: int) -> bytes:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {vertex_count}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property float uncertainty",
            "property int anchor_id",
            "property short level",
            "property int point_count",
            "property uchar is_observed",
            "property float cell_size",
            "end_header",
            "",
        ]
    )
    return header.encode("ascii")


def write_binary_ply(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(ply_header(int(rows.shape[0])))
        rows.astype(PLY_DTYPE, copy=False).tofile(handle)


def load_selected_frames(path: Path) -> list[int]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    selected = metadata.get("selected_frames")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"`selected_frames` is missing or empty in {path}")
    return [int(frame_id) for frame_id in selected]


def parse_cam0_to_world(path: Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = line.strip().split()
            if not tokens:
                continue
            frame_id = int(tokens[0])
            values = np.asarray([float(token) for token in tokens[1:]], dtype=np.float64)
            if values.size != 16:
                raise ValueError(
                    f"Expected 16 matrix values for frame {frame_id}, found {values.size}"
                )
            poses[frame_id] = values.reshape(4, 4)
    if not poses:
        raise ValueError(f"No poses found in {path}")
    return poses


def build_trajectory_rows(
    frame_ids: list[int],
    poses_by_frame: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    missing = [frame_id for frame_id in frame_ids if frame_id not in poses_by_frame]
    if missing:
        preview = ", ".join(str(frame_id) for frame_id in missing[:5])
        raise KeyError(f"Missing poses for selected frame ids: {preview}")

    centers = np.stack([poses_by_frame[frame_id][:3, 3] for frame_id in frame_ids], axis=0)
    vertex_rows = np.empty(len(frame_ids), dtype=TRAJECTORY_VERTEX_DTYPE)
    vertex_rows["x"] = centers[:, 0].astype(np.float32)
    vertex_rows["y"] = centers[:, 1].astype(np.float32)
    vertex_rows["z"] = centers[:, 2].astype(np.float32)
    vertex_rows["red"] = np.uint8(255)
    vertex_rows["green"] = np.uint8(255)
    vertex_rows["blue"] = np.uint8(0)
    vertex_rows["frame_id"] = np.asarray(frame_ids, dtype=np.int32)
    vertex_rows["trajectory_index"] = np.arange(len(frame_ids), dtype=np.int32)

    edge_count = max(len(frame_ids) - 1, 0)
    edge_rows = np.empty(edge_count, dtype=TRAJECTORY_EDGE_DTYPE)
    if edge_count:
        edge_rows["vertex1"] = np.arange(edge_count, dtype=np.int32)
        edge_rows["vertex2"] = np.arange(1, edge_count + 1, dtype=np.int32)
    return vertex_rows, edge_rows


def trajectory_ply_header(vertex_count: int, edge_count: int) -> bytes:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {vertex_count}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int frame_id",
            "property int trajectory_index",
            f"element edge {edge_count}",
            "property int vertex1",
            "property int vertex2",
            "end_header",
            "",
        ]
    )
    return header.encode("ascii")


def write_trajectory_ply(path: Path, vertex_rows: np.ndarray, edge_rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(trajectory_ply_header(int(vertex_rows.shape[0]), int(edge_rows.shape[0])))
        vertex_rows.astype(TRAJECTORY_VERTEX_DTYPE, copy=False).tofile(handle)
        edge_rows.astype(TRAJECTORY_EDGE_DTYPE, copy=False).tofile(handle)


def export_camera_trajectory(
    *,
    drive: str,
    output_dir: Path,
    selection_metadata: Path,
    poses_root: Path | None,
) -> dict[str, Any]:
    resolved_poses_root = resolve_kitti360_path(poses_root, kind="poses")
    poses_path = resolved_poses_root / drive / "cam0_to_world.txt"
    frame_ids = load_selected_frames(selection_metadata)
    poses_by_frame = parse_cam0_to_world(poses_path)
    vertex_rows, edge_rows = build_trajectory_rows(frame_ids, poses_by_frame)

    trajectory_path = output_dir / "camera_trajectory.ply"
    write_trajectory_ply(trajectory_path, vertex_rows, edge_rows)
    return {
        "trajectory_path": str(trajectory_path),
        "selection_metadata": str(selection_metadata),
        "poses_path": str(poses_path),
        "camera_count": int(vertex_rows.shape[0]),
        "edge_count": int(edge_rows.shape[0]),
        "first_frame_id": int(frame_ids[0]),
        "last_frame_id": int(frame_ids[-1]),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p98": 0.0, "max": 0.0}
    return {
        "min": float(np.min(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p98": float(np.percentile(finite, 98)),
        "max": float(np.max(finite)),
    }


def export_uncertainty_map(
    *,
    bucket_root: Path,
    uncertainty_path: Path,
    posterior_path: Path,
    output_dir: Path,
    vmin: float | None,
    vmax: float | None,
    percentile_low: float,
    percentile_high: float,
    observed_only: bool,
    split_levels: bool,
    trajectory_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pts_path = bucket_root / "pts_by_anchor.npz"
    if not pts_path.exists():
        raise FileNotFoundError(f"Could not find anchor artifact: {pts_path}")

    pts_by_anchor = np.load(pts_path)
    posterior = np.load(posterior_path)
    uncertainty = np.asarray(np.load(uncertainty_path), dtype=np.float32)

    anchor_xyz = np.asarray(pts_by_anchor["anchor_xyz"], dtype=np.float32)
    anchor_level = np.asarray(pts_by_anchor["anchor_level"], dtype=np.int16).reshape(-1)
    point_counts = np.asarray(pts_by_anchor["point_counts"], dtype=np.int32).reshape(-1)
    is_observed = np.asarray(posterior["is_observed"], dtype=bool).reshape(-1)

    anchor_count = int(anchor_xyz.shape[0])
    for name, values in (
        ("anchor_level", anchor_level),
        ("point_counts", point_counts),
        ("uncertainty", uncertainty),
        ("is_observed", is_observed),
    ):
        if values.shape[0] != anchor_count:
            raise ValueError(
                f"`{name}` length {values.shape[0]} does not match anchor count {anchor_count}"
            )

    voxel_size = _scalar_item(pts_by_anchor["voxel_size"], "voxel_size")
    fork = _scalar_item(pts_by_anchor["fork"], "fork")
    levels = int(_scalar_item(pts_by_anchor["levels"], "levels"))
    cell_size = (voxel_size / np.power(fork, anchor_level.astype(np.float32))).astype(np.float32)

    scale_vmin, scale_vmax, scale_source = choose_color_scale(
        uncertainty,
        is_observed,
        vmin=vmin,
        vmax=vmax,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    rgb = uncertainty_to_rgb(
        uncertainty,
        is_observed,
        vmin=scale_vmin,
        vmax=scale_vmax,
    )

    if observed_only:
        export_mask = is_observed
    else:
        export_mask = np.ones(anchor_count, dtype=bool)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_path = output_dir / "anchors_uncertainty_all.ply"
    all_rows = build_vertex_array(
        anchor_xyz=anchor_xyz[export_mask],
        anchor_level=anchor_level[export_mask],
        point_counts=point_counts[export_mask],
        uncertainty=uncertainty[export_mask],
        is_observed=is_observed[export_mask],
        cell_size=cell_size[export_mask],
        rgb=rgb[export_mask],
        anchor_ids=np.nonzero(export_mask)[0].astype(np.int32),
    )
    write_binary_ply(all_path, all_rows)

    level_paths: list[str] = []
    level_counts: dict[str, int] = {}
    if split_levels:
        for level in range(levels):
            level_mask = export_mask & (anchor_level == level)
            level_rows = build_vertex_array(
                anchor_xyz=anchor_xyz[level_mask],
                anchor_level=anchor_level[level_mask],
                point_counts=point_counts[level_mask],
                uncertainty=uncertainty[level_mask],
                is_observed=is_observed[level_mask],
                cell_size=cell_size[level_mask],
                rgb=rgb[level_mask],
                anchor_ids=np.nonzero(level_mask)[0].astype(np.int32),
            )
            level_path = output_dir / f"anchors_uncertainty_lod_{level:02d}.ply"
            write_binary_ply(level_path, level_rows)
            level_paths.append(str(level_path))
            level_counts[str(level)] = int(level_rows.shape[0])

    metadata = {
        "bucket_root": str(bucket_root),
        "pts_by_anchor_path": str(pts_path),
        "uncertainty_path": str(uncertainty_path),
        "posterior_path": str(posterior_path),
        "output_dir": str(output_dir),
        "all_path": str(all_path),
        "level_paths": level_paths,
        "anchor_count": anchor_count,
        "exported_anchor_count": int(all_rows.shape[0]),
        "observed_anchor_count": int(np.count_nonzero(is_observed)),
        "unobserved_anchor_count": int(anchor_count - np.count_nonzero(is_observed)),
        "observed_only": bool(observed_only),
        "split_levels": bool(split_levels),
        "levels": levels,
        "level_counts": level_counts,
        "trajectory": trajectory_metadata,
        "voxel_size": voxel_size,
        "fork": fork,
        "color_scale": {
            "vmin": scale_vmin,
            "vmax": scale_vmax,
            "source": scale_source,
            "percentile_low": percentile_low,
            "percentile_high": percentile_high,
            "unobserved_color": [255, 0, 0],
        },
        "all_summary": summarize(uncertainty),
        "observed_summary": summarize(uncertainty[is_observed]),
    }
    save_json(output_dir / "uncertainty_map_metadata.json", metadata)
    return metadata


def main() -> None:
    args = parse_args()
    bucket_root = resolve_bucket_root(args.drive, args.bucket_root)
    uncertainty_path = resolve_uncertainty_path(bucket_root, args.uncertainty)
    posterior_path = resolve_posterior_path(bucket_root, args.posterior)
    output_dir = resolve_output_dir(args.drive, args.output_dir)
    selection_metadata = resolve_selection_metadata(args.drive, args.selection_metadata)

    trajectory_metadata = None
    if not args.no_trajectory:
        if selection_metadata.exists():
            print(f"Loading selected camera frames: {selection_metadata}")
            trajectory_metadata = export_camera_trajectory(
                drive=args.drive,
                output_dir=output_dir,
                selection_metadata=selection_metadata,
                poses_root=args.poses_root,
            )
            print(f"Wrote {trajectory_metadata['trajectory_path']}")
        else:
            print(
                "Skipping camera trajectory: "
                f"selection metadata not found at {selection_metadata}"
            )

    print(f"Loading anchors: {bucket_root / 'pts_by_anchor.npz'}")
    print(f"Loading uncertainty: {uncertainty_path}")
    print(f"Loading posterior mask: {posterior_path}")
    metadata = export_uncertainty_map(
        bucket_root=bucket_root,
        uncertainty_path=uncertainty_path,
        posterior_path=posterior_path,
        output_dir=output_dir,
        vmin=args.vmin,
        vmax=args.vmax,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        observed_only=args.observed_only,
        split_levels=not args.no_split_levels,
        trajectory_metadata=trajectory_metadata,
    )

    print(f"Wrote {metadata['all_path']}")
    for level_path in metadata["level_paths"]:
        print(f"Wrote {level_path}")
    print(f"Wrote {output_dir / 'uncertainty_map_metadata.json'}")
    scale = metadata["color_scale"]
    print(
        "Color scale: "
        f"vmin={scale['vmin']:.6g} vmax={scale['vmax']:.6g} source={scale['source']}"
    )


if __name__ == "__main__":
    main()
