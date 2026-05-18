#!/usr/bin/env python3

"""Train the original global VBGS model on KITTI-360 point artifacts.

This is a repo-owned baseline wrapper around the upstream ``vbgs`` training
APIs. It intentionally trains one scene-wide DeltaMixture, rather than the
per-anchor posteriors used by VBOGS.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bucket_points import (
    normalized_points_from_xyz_rgb,
    normalization_params_from_xyz_rgb,
    select_points,
)
from vbogs.io import save_json


DEFAULT_BUCKET_ROOT = REPO_ROOT / "data" / "m4"
DEFAULT_POINTS_ROOT = REPO_ROOT / "data" / "points_world"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "vbgs_baseline"


@dataclass(frozen=True)
class BaselineInput:
    points_norm: np.ndarray
    norm_params: dict[str, np.ndarray]
    input_mode: str
    normalization_source: str
    source_paths: dict[str, str | None]
    source_point_count: int
    selected_point_count: int
    frame_count: int
    point_selection: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive",
        default="2013_05_28_drive_0007_sync",
        help="KITTI-360 drive id used to resolve default input/output paths.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("auto", "bucket", "stereo"),
        default="auto",
        help=(
            "`auto` prefers M4 bucket artifacts, then falls back to stereo "
            "point-cloud artifacts."
        ),
    )
    parser.add_argument(
        "--bucket-root",
        type=Path,
        default=None,
        help="Root containing `data/m4/<drive>` artifacts.",
    )
    parser.add_argument(
        "--points-norm",
        type=Path,
        default=None,
        help="Explicit M4 `points_norm.npz` path.",
    )
    parser.add_argument(
        "--norm-params",
        type=Path,
        default=None,
        help="Explicit M4 `norm_params.json` path.",
    )
    parser.add_argument(
        "--points-world",
        type=Path,
        default=None,
        help="Explicit stereo `points_world.npz` path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to `outputs/vbgs_baseline/<drive>`.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help=(
            "Optional deterministic point cap for `--input-mode stereo`; "
            "`0` keeps all stereo points."
        ),
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=10_000,
        help="Number of global VBGS mixture components.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="VBGS sufficient-stat batch size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed for initialization and reassignment.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Index into `jax.devices()` used for training.",
    )
    parser.add_argument(
        "--reassign-fraction",
        type=float,
        default=0.05,
        help="Fraction of unused components to reassign before fitting.",
    )
    parser.add_argument(
        "--no-reassign",
        action="store_true",
        help="Disable the upstream Habitat-style reassignment heuristic.",
    )
    parser.add_argument(
        "--vbgs-root",
        type=Path,
        default=Path("vbgs"),
        help="Path to the vbgs submodule.",
    )
    return parser.parse_args(argv)


def resolve_bucket_root(drive: str, bucket_root: Path | None) -> Path:
    if bucket_root is not None:
        return bucket_root.resolve()
    return (DEFAULT_BUCKET_ROOT / drive).resolve()


def resolve_output_root(drive: str, output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root.resolve()
    return (DEFAULT_OUTPUT_ROOT / drive).resolve()


def default_points_norm_path(drive: str, bucket_root: Path | None) -> Path:
    return resolve_bucket_root(drive, bucket_root) / "points_norm.npz"


def default_norm_params_path(drive: str, bucket_root: Path | None) -> Path:
    return resolve_bucket_root(drive, bucket_root) / "norm_params.json"


def default_points_world_path(drive: str) -> Path:
    return (DEFAULT_POINTS_ROOT / drive / "points_world.npz").resolve()


def load_norm_params(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        "offset": np.asarray(raw["offset"], dtype=np.float32),
        "stdevs": np.asarray(raw["stdevs"], dtype=np.float32),
    }


def save_norm_params(path: Path, norm_params: dict[str, np.ndarray]) -> None:
    save_json(
        path,
        {
            "offset": np.asarray(norm_params["offset"], dtype=np.float32).tolist(),
            "stdevs": np.asarray(norm_params["stdevs"], dtype=np.float32).tolist(),
        },
    )


def frame_count_from_ids(frame_id: np.ndarray | None) -> int:
    if frame_id is None:
        return 0
    frame_id = np.asarray(frame_id)
    if frame_id.size == 0:
        return 0
    return int(np.unique(frame_id).shape[0])


def bucket_artifacts_exist(
    *,
    drive: str,
    bucket_root: Path | None,
    points_norm: Path | None,
    norm_params: Path | None,
) -> bool:
    points_norm_path = (points_norm or default_points_norm_path(drive, bucket_root)).resolve()
    norm_params_path = (norm_params or default_norm_params_path(drive, bucket_root)).resolve()
    return points_norm_path.exists() and norm_params_path.exists()


def load_bucket_input(
    *,
    drive: str,
    bucket_root: Path | None,
    points_norm: Path | None,
    norm_params: Path | None,
) -> BaselineInput:
    points_norm_path = (points_norm or default_points_norm_path(drive, bucket_root)).resolve()
    norm_params_path = (norm_params or default_norm_params_path(drive, bucket_root)).resolve()

    if not points_norm_path.exists():
        raise FileNotFoundError(f"M4 points artifact not found: {points_norm_path}")
    if not norm_params_path.exists():
        raise FileNotFoundError(f"M4 normalization params not found: {norm_params_path}")

    with np.load(points_norm_path) as data:
        points = np.asarray(data["points_norm"], dtype=np.float32)
        frame_id = np.asarray(data["frame_id"], dtype=np.int32) if "frame_id" in data else None

    return BaselineInput(
        points_norm=points,
        norm_params=load_norm_params(norm_params_path),
        input_mode="bucket",
        normalization_source="copied",
        source_paths={
            "points_norm": str(points_norm_path),
            "norm_params": str(norm_params_path),
            "points_world": None,
        },
        source_point_count=int(points.shape[0]),
        selected_point_count=int(points.shape[0]),
        frame_count=frame_count_from_ids(frame_id),
        point_selection={
            "source_point_count": int(points.shape[0]),
            "selected_point_count": int(points.shape[0]),
            "max_points": 0,
            "selection": "bucket_exact",
        },
    )


def load_stereo_input(
    *,
    drive: str,
    points_world: Path | None,
    max_points: int,
) -> BaselineInput:
    points_world_path = (points_world or default_points_world_path(drive)).resolve()
    if not points_world_path.exists():
        raise FileNotFoundError(f"Stereo point artifact not found: {points_world_path}")

    with np.load(points_world_path) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float32)
        rgb = np.asarray(data["rgb"], dtype=np.uint8)
        frame_id = (
            np.asarray(data["frame_id"], dtype=np.int32)
            if "frame_id" in data
            else np.zeros((xyz.shape[0],), dtype=np.int32)
        )

    selected_xyz, selected_rgb, selected_frame_id, point_selection = select_points(
        xyz,
        rgb,
        frame_id,
        max_points=max_points,
    )
    norm_params = normalization_params_from_xyz_rgb(selected_xyz, selected_rgb)
    points = normalized_points_from_xyz_rgb(
        selected_xyz,
        selected_rgb,
        norm_params,
        chunk_size=1_000_000,
    )

    return BaselineInput(
        points_norm=points,
        norm_params=norm_params,
        input_mode="stereo",
        normalization_source="generated",
        source_paths={
            "points_norm": None,
            "norm_params": None,
            "points_world": str(points_world_path),
        },
        source_point_count=int(point_selection["source_point_count"]),
        selected_point_count=int(point_selection["selected_point_count"]),
        frame_count=frame_count_from_ids(selected_frame_id),
        point_selection=point_selection,
    )


def load_baseline_input(args: argparse.Namespace) -> BaselineInput:
    if args.input_mode == "bucket":
        return load_bucket_input(
            drive=args.drive,
            bucket_root=args.bucket_root,
            points_norm=args.points_norm,
            norm_params=args.norm_params,
        )
    if args.input_mode == "stereo":
        return load_stereo_input(
            drive=args.drive,
            points_world=args.points_world,
            max_points=args.max_points,
        )
    if bucket_artifacts_exist(
        drive=args.drive,
        bucket_root=args.bucket_root,
        points_norm=args.points_norm,
        norm_params=args.norm_params,
    ):
        return load_bucket_input(
            drive=args.drive,
            bucket_root=args.bucket_root,
            points_norm=args.points_norm,
            norm_params=args.norm_params,
        )
    return load_stereo_input(
        drive=args.drive,
        points_world=args.points_world,
        max_points=args.max_points,
    )


def add_vbgs_to_path(vbgs_root: Path) -> None:
    vbgs_root = vbgs_root.resolve()
    if str(vbgs_root) not in sys.path:
        sys.path.insert(0, str(vbgs_root))
    scripts_dir = vbgs_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def validate_training_args(args: argparse.Namespace, data: BaselineInput) -> None:
    if args.n_components <= 0:
        raise ValueError("--n-components must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_points < 0:
        raise ValueError("--max-points must be non-negative")
    if args.reassign_fraction < 0.0 or args.reassign_fraction > 1.0:
        raise ValueError("--reassign-fraction must be in [0, 1]")
    if data.points_norm.ndim != 2 or data.points_norm.shape[1] != 6:
        raise ValueError(
            "`points_norm` must have shape (N, 6); "
            f"got {data.points_norm.shape}"
        )
    if data.points_norm.shape[0] == 0:
        raise ValueError("Cannot train a VBGS baseline with zero points")


def select_device(jax_module: Any, device_index: int) -> str:
    devices = jax_module.devices()
    if not devices:
        raise RuntimeError("JAX reported no available devices")
    if device_index < 0 or device_index >= len(devices):
        raise ValueError(
            f"Requested JAX device {device_index}, but only "
            f"{len(devices)} device(s) exist"
        )
    device = devices[device_index]
    jax_module.config.update("jax_default_device", device)
    return str(device)


def compute_mean_elbo(
    *,
    model: Any,
    points_norm: np.ndarray,
    batch_size: int,
    compute_elbo_delta: Any,
    jnp_module: Any,
) -> float:
    total = 0.0
    count = 0
    for start in range(0, points_norm.shape[0], batch_size):
        batch = points_norm[start : start + batch_size]
        batch_jax = jnp_module.expand_dims(jnp_module.asarray(batch), axis=-1)
        elbo_arr, _ = compute_elbo_delta(model, batch_jax)
        total += float(np.asarray(elbo_arr).sum())
        count += int(batch.shape[0])
    return total / max(count, 1)


def maybe_reassign_prior(
    *,
    prior_model: Any,
    model: Any,
    points_norm: np.ndarray,
    batch_size: int,
    fraction: float,
    reassign_fn: Any,
) -> tuple[Any, bool, int]:
    if fraction <= 0.0:
        return prior_model, False, 0
    initial_alpha = np.asarray(prior_model.prior.prior_alpha)
    current_alpha = np.asarray(model.prior.alpha)
    available = int(np.count_nonzero(current_alpha <= initial_alpha.min()))
    n_reassign = int(available * fraction)
    if n_reassign <= 0:
        return prior_model, False, 0
    if n_reassign >= points_norm.shape[0]:
        print(
            "Skipping reassignment because the requested reassignment count "
            f"({n_reassign:,}) is not smaller than the point count "
            f"({points_norm.shape[0]:,})."
        )
        return prior_model, False, n_reassign
    return (
        reassign_fn(prior_model, model, points_norm, batch_size, fraction),
        True,
        n_reassign,
    )


def train_baseline(args: argparse.Namespace, data: BaselineInput) -> dict[str, Any]:
    add_vbgs_to_path(args.vbgs_root)

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from model_volume import get_volume_delta_mixture
    from vbgs.model.reassign import reassign
    from vbgs.model.train import compute_elbo_delta, fit_gmm_step
    from vbgs.model.utils import random_mean_init, store_model

    device_name = select_device(jax, args.device)
    np.random.seed(args.seed)

    output_root = resolve_output_root(args.drive, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    key = jr.PRNGKey(args.seed)
    key, subkey = jr.split(key)
    mean_init = random_mean_init(
        key=subkey,
        x=None,
        component_shape=(args.n_components,),
        event_shape=(6, 1),
        init_random=True,
        add_noise=True,
    )

    key, subkey = jr.split(key)
    prior_model = get_volume_delta_mixture(
        key=subkey,
        n_components=args.n_components,
        mean_init=mean_init,
        beta=0,
        learning_rate=1,
        dof_offset=1,
        position_scale=args.n_components,
        position_event_shape=(3, 1),
    )
    model = copy.deepcopy(prior_model)

    fit_start = time.time()
    reassign_applied = False
    n_reassign = 0
    if not args.no_reassign:
        prior_model, reassign_applied, n_reassign = maybe_reassign_prior(
            prior_model=prior_model,
            model=model,
            points_norm=data.points_norm,
            batch_size=args.batch_size,
            fraction=args.reassign_fraction,
            reassign_fn=reassign,
        )

    model, _prior_stats, _space_stats, _color_stats = fit_gmm_step(
        prior_model,
        model,
        data=data.points_norm,
        batch_size=args.batch_size,
        prior_stats=None,
        space_stats=None,
        color_stats=None,
    )
    mean_elbo = compute_mean_elbo(
        model=model,
        points_norm=data.points_norm,
        batch_size=args.batch_size,
        compute_elbo_delta=compute_elbo_delta,
        jnp_module=jnp,
    )
    elapsed = time.time() - fit_start

    model_json_path = output_root / "model_final.json"
    model_npz_path = output_root / "model_final.npz"
    metadata_path = output_root / "baseline_metadata.json"
    norm_params_path = output_root / "normalization_params.json"

    store_model(model, data.norm_params, str(model_json_path))
    mu, sigma = model.denormalize(data.norm_params, clip_val=None)
    alpha = model.prior.alpha.reshape(-1)
    np.savez_compressed(
        model_npz_path,
        mu=np.asarray(mu, dtype=np.float32),
        sigma=np.asarray(sigma, dtype=np.float32),
        alpha=np.asarray(alpha, dtype=np.float32),
    )
    save_norm_params(norm_params_path, data.norm_params)

    metadata = {
        "drive": args.drive,
        "input_mode": data.input_mode,
        "source_paths": data.source_paths,
        "normalization_source": data.normalization_source,
        "normalization_params_path": str(norm_params_path),
        "source_point_count": int(data.source_point_count),
        "selected_point_count": int(data.selected_point_count),
        "frame_count": int(data.frame_count),
        "point_selection": data.point_selection,
        "n_components": int(args.n_components),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "jax_device": device_name,
        "reassign_enabled": not args.no_reassign,
        "reassign_fraction": float(args.reassign_fraction),
        "reassign_applied": bool(reassign_applied),
        "reassign_requested_count": int(n_reassign),
        "elapsed_sec": float(elapsed),
        "points_per_sec": float(data.selected_point_count / max(elapsed, 1e-6)),
        "mean_elbo_per_point": float(mean_elbo),
        "artifacts": {
            "model_json": str(model_json_path),
            "model_npz": str(model_npz_path),
            "metadata": str(metadata_path),
        },
    }
    save_json(metadata_path, metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data = load_baseline_input(args)
    validate_training_args(args, data)
    metadata = train_baseline(args, data)

    print(f"Wrote {metadata['artifacts']['model_json']}")
    print(f"Wrote {metadata['artifacts']['model_npz']}")
    print(f"Wrote {metadata['artifacts']['metadata']}")
    print(
        "Baseline summary: "
        f"points={metadata['selected_point_count']:,} "
        f"K={metadata['n_components']:,} "
        f"ELBO/pt={metadata['mean_elbo_per_point']:.4f}"
    )


if __name__ == "__main__":
    main()
