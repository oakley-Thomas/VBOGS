#!/usr/bin/env python3

"""Fit per-anchor VBGS posteriors from packed M4a artifacts.

This is M4b from PLAN.md. It reads:

- `points_norm.npz`
- `pts_by_anchor.npz`

and writes:

- `anchor_posterior.npz`
- `fit_metadata.json`
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.io import save_json, unpack_group_slice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive",
        default="2013_05_28_drive_0008_sync",
        help="Drive id used to resolve default input/output paths.",
    )
    parser.add_argument(
        "--bucket-root",
        type=Path,
        default=None,
        help="Directory containing the M4a outputs. Defaults to `data/m4/<drive>`.",
    )
    parser.add_argument(
        "--points-norm",
        type=Path,
        default=None,
        help="Explicit path to `points_norm.npz`.",
    )
    parser.add_argument(
        "--pts-by-anchor",
        type=Path,
        default=None,
        help="Explicit path to `pts_by_anchor.npz`.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where `anchor_posterior.npz` will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base PRNG seed for per-anchor model initialization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Maximum VBEM sufficient-stat batch size.",
    )
    parser.add_argument(
        "--batch-buckets",
        default="64,128,256,512,1024,2048,4096,5000",
        help=(
            "Comma-separated padded batch sizes used when --adaptive-batch-size is enabled. "
            "Small anchors use the smallest bucket that fits their point count, capped by --batch-size."
        ),
    )
    parser.add_argument(
        "--no-adaptive-batch-size",
        action="store_true",
        help="Disable per-anchor batch-size buckets and always use --batch-size.",
    )
    parser.add_argument(
        "--k-init",
        type=int,
        default=10,
        help="Initial number of mixture components.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=40,
        help="Maximum number of mixture components.",
    )
    parser.add_argument(
        "--k-growth-factor",
        type=int,
        default=2,
        help="Multiplicative K growth factor.",
    )
    parser.add_argument(
        "--k-growth-min-points",
        type=int,
        default=0,
        help=(
            "Only try larger K values for anchors with at least this many points. "
            "Default 0 preserves the PLAN.md K-growth behavior for every observed anchor."
        ),
    )
    parser.add_argument(
        "--min-points-per-anchor",
        type=int,
        default=20,
        help="Anchors with fewer points are marked unobserved.",
    )
    parser.add_argument(
        "--elbo-improvement-tol",
        type=float,
        default=0.01,
        help="Minimum mean-ELBO gain required to accept a larger K.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print a progress line every N observed anchors.",
    )
    parser.add_argument(
        "--max-observed-anchors",
        type=int,
        default=0,
        help="Optional cap for smoke tests; 0 processes all observed anchors.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Write a resumable checkpoint every N newly fitted anchors; 0 disables checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint file for this output if it exists.",
    )
    parser.add_argument(
        "--vbgs-root",
        type=Path,
        default=Path("vbgs"),
        help="Path to the vbgs submodule.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Index into `jax.devices()` used for the fit.",
    )
    return parser.parse_args()


def resolve_bucket_root(args: argparse.Namespace) -> Path:
    if args.bucket_root is not None:
        return args.bucket_root
    return Path("data/m4") / args.drive


def resolve_output_root(args: argparse.Namespace, bucket_root: Path) -> Path:
    if args.output_root is not None:
        return args.output_root
    return bucket_root


def resolve_output_paths(output_root: Path, max_observed_anchors: int) -> Tuple[Path, Path]:
    if max_observed_anchors > 0:
        return (
            output_root / "anchor_posterior.smoke.npz",
            output_root / "fit_metadata.smoke.json",
        )
    return (
        output_root / "anchor_posterior.npz",
        output_root / "fit_metadata.json",
    )


def resolve_checkpoint_paths(posterior_path: Path, metadata_path: Path) -> Tuple[Path, Path]:
    return (
        posterior_path.with_name(f"{posterior_path.stem}.checkpoint.npz"),
        metadata_path.with_name(f"{metadata_path.stem}.checkpoint.json"),
    )


def add_vbgs_to_path(vbgs_root: Path) -> None:
    vbgs_root = vbgs_root.resolve()
    if str(vbgs_root) not in sys.path:
        sys.path.insert(0, str(vbgs_root))
    scripts_dir = vbgs_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def select_device(device_index: int) -> str:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported no available devices")
    if device_index < 0 or device_index >= len(devices):
        raise ValueError(f"Requested JAX device {device_index}, but only {len(devices)} device(s) exist")
    device = devices[device_index]
    jax.config.update("jax_default_device", device)
    return str(device)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp_path.replace(path)


def compute_mean_elbo(model, points_norm: np.ndarray, batch_size: int, compute_elbo_delta) -> float:
    total = 0.0
    count = 0
    for start in range(0, points_norm.shape[0], batch_size):
        batch = points_norm[start : start + batch_size]
        valid_count = int(batch.shape[0])
        if valid_count < batch_size:
            padded = np.zeros((batch_size, points_norm.shape[1]), dtype=points_norm.dtype)
            padded[:valid_count] = batch
            batch = padded
        batch_jax = jnp.expand_dims(jnp.asarray(batch), axis=-1)
        elbo_arr, _ = compute_elbo_delta(model, batch_jax)
        total += float(np.asarray(elbo_arr)[:valid_count].sum())
        count += valid_count
    return total / max(count, 1)


def parse_batch_buckets(raw_buckets: str, max_batch_size: int) -> Tuple[int, ...]:
    if max_batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    buckets = []
    for raw_bucket in raw_buckets.split(","):
        raw_bucket = raw_bucket.strip()
        if not raw_bucket:
            continue
        bucket = int(raw_bucket)
        if bucket <= 0:
            raise ValueError("--batch-buckets values must be positive integers")
        if bucket <= max_batch_size:
            buckets.append(bucket)
    buckets.append(max_batch_size)
    return tuple(sorted(set(buckets)))


def choose_batch_size(point_count: int, max_batch_size: int, batch_buckets: Tuple[int, ...], adaptive: bool) -> int:
    if not adaptive:
        return max_batch_size
    target = min(max(point_count, 1), max_batch_size)
    for bucket in batch_buckets:
        if bucket >= target:
            return bucket
    return max_batch_size


def fit_anchor(
    points_norm: np.ndarray,
    n_components: int,
    *,
    seed: int,
    batch_size: int,
    random_mean_init,
    fit_gmm_step,
    compute_elbo_delta,
    get_volume_delta_mixture,
):
    key = jr.PRNGKey(seed)
    key, subkey = jr.split(key)
    mean_init = random_mean_init(
        key=subkey,
        x=jnp.asarray(points_norm),
        component_shape=(n_components,),
        event_shape=(points_norm.shape[1], 1),
        init_random=False,
        add_noise=True,
    )

    key, subkey = jr.split(key)
    prior_model = get_volume_delta_mixture(
        key=subkey,
        n_components=n_components,
        mean_init=mean_init,
        position_event_shape=(3, 1),
        beta=0,
        learning_rate=1,
    )
    model = copy.deepcopy(prior_model)

    prior_stats = None
    space_stats = None
    color_stats = None
    model, prior_stats, space_stats, color_stats = fit_gmm_step(
        prior_model,
        model,
        data=points_norm,
        batch_size=batch_size,
        prior_stats=prior_stats,
        space_stats=space_stats,
        color_stats=color_stats,
    )
    elbo_per_point = compute_mean_elbo(model, points_norm, batch_size, compute_elbo_delta)
    return model, float(elbo_per_point)


def pack_model_params(model, k_max: int) -> Dict[str, np.ndarray]:
    alpha = np.full((k_max,), np.nan, dtype=np.float32)
    spatial_mean = np.full((k_max, 3, 1), np.nan, dtype=np.float32)
    spatial_kappa = np.full((k_max, 1, 1), np.nan, dtype=np.float32)
    spatial_u = np.full((k_max, 3, 3), np.nan, dtype=np.float32)
    spatial_n = np.full((k_max, 1, 1), np.nan, dtype=np.float32)
    delta_mean = np.full((k_max, 3, 1), np.nan, dtype=np.float32)
    delta_kappa = np.full((k_max, 1, 1), np.nan, dtype=np.float32)
    delta_u = np.full((k_max, 3, 3), np.nan, dtype=np.float32)
    delta_n = np.full((k_max, 1, 1), np.nan, dtype=np.float32)

    k = int(np.asarray(model.mixture.prior.alpha).shape[0])
    alpha[:k] = np.asarray(model.mixture.prior.alpha, dtype=np.float32)
    spatial_mean[:k] = np.asarray(model.mixture.likelihood.mean, dtype=np.float32)
    spatial_kappa[:k] = np.asarray(model.mixture.likelihood.kappa, dtype=np.float32)
    spatial_u[:k] = np.asarray(model.mixture.likelihood.u, dtype=np.float32)
    spatial_n[:k] = np.asarray(model.mixture.likelihood.n, dtype=np.float32)
    delta_mean[:k] = np.asarray(model.delta.mean, dtype=np.float32)
    delta_kappa[:k] = np.asarray(model.delta.kappa, dtype=np.float32)
    delta_u[:k] = np.asarray(model.delta.u, dtype=np.float32)
    delta_n[:k] = np.asarray(model.delta.n, dtype=np.float32)

    return {
        "alpha": alpha,
        "spatial_mean": spatial_mean,
        "spatial_kappa": spatial_kappa,
        "spatial_u": spatial_u,
        "spatial_n": spatial_n,
        "delta_mean": delta_mean,
        "delta_kappa": delta_kappa,
        "delta_u": delta_u,
        "delta_n": delta_n,
        "k": k,
    }


def build_metadata(
    args: argparse.Namespace,
    *,
    device_name: str,
    points_norm_path: Path,
    pts_by_anchor_path: Path,
    n_anchors: int,
    m_obs: int,
    total_observed: int,
    observed_point_counts: np.ndarray,
    k_growth_attempted: np.ndarray,
    fit_batch_size: np.ndarray,
    under_modeled: np.ndarray,
    fit_completed: np.ndarray,
    adaptive_batch_size: bool,
    batch_buckets: Tuple[int, ...],
    elapsed: float,
    run_completed_count: int,
    checkpoint_path: Path | None = None,
) -> Dict:
    completed_count = int(fit_completed.sum())
    return {
        "drive": args.drive,
        "jax_device": device_name,
        "points_norm_path": str(points_norm_path),
        "pts_by_anchor_path": str(pts_by_anchor_path),
        "anchor_count": n_anchors,
        "observed_anchor_count": int(m_obs),
        "unobserved_anchor_count": int(n_anchors - total_observed),
        "observed_anchor_count_total": total_observed,
        "completed_anchor_count": completed_count,
        "remaining_anchor_count": int(m_obs - completed_count),
        "run_completed_anchor_count": int(run_completed_count),
        "observed_point_count_min": int(observed_point_counts.min()) if m_obs else 0,
        "observed_point_count_p50": float(np.percentile(observed_point_counts, 50)) if m_obs else 0.0,
        "observed_point_count_p90": float(np.percentile(observed_point_counts, 90)) if m_obs else 0.0,
        "observed_point_count_max": int(observed_point_counts.max()) if m_obs else 0,
        "k_init": args.k_init,
        "k_max": args.k_max,
        "k_growth_factor": args.k_growth_factor,
        "k_growth_min_points": args.k_growth_min_points,
        "k_growth_attempted_count": int(k_growth_attempted.sum()),
        "batch_size": args.batch_size,
        "adaptive_batch_size": adaptive_batch_size,
        "batch_buckets": list(batch_buckets),
        "fit_batch_size_min": int(fit_batch_size[fit_completed].min()) if completed_count else 0,
        "fit_batch_size_p50": float(np.percentile(fit_batch_size[fit_completed], 50)) if completed_count else 0.0,
        "fit_batch_size_p90": float(np.percentile(fit_batch_size[fit_completed], 90)) if completed_count else 0.0,
        "fit_batch_size_max": int(fit_batch_size[fit_completed].max()) if completed_count else 0,
        "min_points_per_anchor": args.min_points_per_anchor,
        "elbo_improvement_tol": args.elbo_improvement_tol,
        "elapsed_sec": elapsed,
        "anchors_per_sec": run_completed_count / max(elapsed, 1e-6),
        "completed_anchors_per_sec": completed_count / max(elapsed, 1e-6),
        "under_modeled_count": int(under_modeled[fit_completed].sum()) if completed_count else 0,
        "max_observed_anchors": args.max_observed_anchors,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }


def posterior_arrays(
    *,
    is_observed: np.ndarray,
    observed_anchor_ids: np.ndarray,
    point_counts: np.ndarray,
    final_k: np.ndarray,
    final_elbo: np.ndarray,
    selected_gain: np.ndarray,
    under_modeled: np.ndarray,
    fit_batch_size: np.ndarray,
    k_growth_attempted: np.ndarray,
    fit_completed: np.ndarray,
    alpha: np.ndarray,
    spatial_mean: np.ndarray,
    spatial_kappa: np.ndarray,
    spatial_u: np.ndarray,
    spatial_n: np.ndarray,
    delta_mean: np.ndarray,
    delta_kappa: np.ndarray,
    delta_u: np.ndarray,
    delta_n: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    return {
        "is_observed": is_observed,
        "observed_anchor_ids": observed_anchor_ids,
        "point_count": point_counts,
        "final_k": final_k,
        "final_elbo": final_elbo,
        "selected_gain": selected_gain,
        "under_modeled": under_modeled,
        "fit_batch_size": fit_batch_size,
        "k_growth_attempted": k_growth_attempted,
        "fit_completed": fit_completed,
        "alpha": alpha,
        "spatial_mean": spatial_mean,
        "spatial_kappa": spatial_kappa,
        "spatial_u": spatial_u,
        "spatial_n": spatial_n,
        "delta_mean": delta_mean,
        "delta_kappa": delta_kappa,
        "delta_u": delta_u,
        "delta_n": delta_n,
        "k_init": np.array(args.k_init, dtype=np.int16),
        "k_max": np.array(args.k_max, dtype=np.int16),
        "min_points_per_anchor": np.array(args.min_points_per_anchor, dtype=np.int32),
        "elbo_improvement_tol": np.array(args.elbo_improvement_tol, dtype=np.float32),
    }


def load_checkpoint(
    checkpoint_path: Path,
    *,
    observed_anchor_ids: np.ndarray,
    final_k: np.ndarray,
    final_elbo: np.ndarray,
    selected_gain: np.ndarray,
    under_modeled: np.ndarray,
    fit_batch_size: np.ndarray,
    k_growth_attempted: np.ndarray,
    fit_completed: np.ndarray,
    alpha: np.ndarray,
    spatial_mean: np.ndarray,
    spatial_kappa: np.ndarray,
    spatial_u: np.ndarray,
    spatial_n: np.ndarray,
    delta_mean: np.ndarray,
    delta_kappa: np.ndarray,
    delta_u: np.ndarray,
    delta_n: np.ndarray,
) -> int:
    checkpoint = load_npz(checkpoint_path)
    checkpoint_anchor_ids = np.asarray(checkpoint["observed_anchor_ids"], dtype=observed_anchor_ids.dtype)
    if checkpoint_anchor_ids.shape != observed_anchor_ids.shape or not np.array_equal(
        checkpoint_anchor_ids, observed_anchor_ids
    ):
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not match the requested observed-anchor set. "
            "Use the same --max-observed-anchors and filtering args, or remove the checkpoint."
        )

    final_k[:] = checkpoint["final_k"]
    final_elbo[:] = checkpoint["final_elbo"]
    selected_gain[:] = checkpoint["selected_gain"]
    under_modeled[:] = checkpoint["under_modeled"]
    fit_batch_size[:] = checkpoint["fit_batch_size"]
    k_growth_attempted[:] = checkpoint["k_growth_attempted"]
    alpha[:] = checkpoint["alpha"]
    spatial_mean[:] = checkpoint["spatial_mean"]
    spatial_kappa[:] = checkpoint["spatial_kappa"]
    spatial_u[:] = checkpoint["spatial_u"]
    spatial_n[:] = checkpoint["spatial_n"]
    delta_mean[:] = checkpoint["delta_mean"]
    delta_kappa[:] = checkpoint["delta_kappa"]
    delta_u[:] = checkpoint["delta_u"]
    delta_n[:] = checkpoint["delta_n"]
    if "fit_completed" in checkpoint:
        fit_completed[:] = checkpoint["fit_completed"]
    else:
        fit_completed[:] = np.isfinite(final_elbo)
    return int(fit_completed.sum())


def main() -> None:
    args = parse_args()
    bucket_root = resolve_bucket_root(args).resolve()
    output_root = resolve_output_root(args, bucket_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    points_norm_path = (args.points_norm or (bucket_root / "points_norm.npz")).resolve()
    pts_by_anchor_path = (args.pts_by_anchor or (bucket_root / "pts_by_anchor.npz")).resolve()
    posterior_path, metadata_path = resolve_output_paths(output_root, args.max_observed_anchors)
    checkpoint_path, checkpoint_metadata_path = resolve_checkpoint_paths(posterior_path, metadata_path)

    add_vbgs_to_path(args.vbgs_root)
    from vbgs.model.train import compute_elbo_delta, fit_gmm_step
    from vbgs.model.utils import random_mean_init
    from model_volume import get_volume_delta_mixture

    device_name = select_device(args.device)
    batch_buckets = parse_batch_buckets(args.batch_buckets, args.batch_size)
    adaptive_batch_size = not args.no_adaptive_batch_size

    points_norm_npz = load_npz(points_norm_path)
    pts_by_anchor_npz = load_npz(pts_by_anchor_path)

    points_norm = np.asarray(points_norm_npz["points_norm"], dtype=np.float32)
    anchor_offsets = np.asarray(pts_by_anchor_npz["anchor_offsets"], dtype=np.int64)
    point_indices = np.asarray(pts_by_anchor_npz["point_indices"], dtype=np.int64)
    point_counts = np.asarray(pts_by_anchor_npz["point_counts"], dtype=np.int32)
    anchor_level = np.asarray(pts_by_anchor_npz["anchor_level"], dtype=np.int16)

    n_anchors = int(anchor_level.shape[0])
    is_observed = point_counts >= args.min_points_per_anchor
    observed_anchor_ids = np.nonzero(is_observed)[0].astype(np.int32)
    if args.max_observed_anchors > 0:
        observed_anchor_ids = observed_anchor_ids[: args.max_observed_anchors]

    print(f"Loaded {points_norm.shape[0]:,} normalized points")
    total_observed = int(is_observed.sum())
    print(
        f"Loaded {n_anchors:,} anchors; {total_observed:,} meet "
        f"MIN_POINTS_PER_ANCHOR={args.min_points_per_anchor}"
    )
    if args.max_observed_anchors > 0:
        print(f"Processing the first {observed_anchor_ids.shape[0]:,} observed anchors for a smoke test")

    m_obs = observed_anchor_ids.shape[0]
    final_k = np.zeros((m_obs,), dtype=np.int16)
    final_elbo = np.full((m_obs,), np.nan, dtype=np.float32)
    selected_gain = np.full((m_obs,), np.nan, dtype=np.float32)
    under_modeled = np.zeros((m_obs,), dtype=bool)
    fit_batch_size = np.zeros((m_obs,), dtype=np.int32)
    k_growth_attempted = np.zeros((m_obs,), dtype=bool)
    fit_completed = np.zeros((m_obs,), dtype=bool)

    alpha = np.full((m_obs, args.k_max), np.nan, dtype=np.float32)
    spatial_mean = np.full((m_obs, args.k_max, 3, 1), np.nan, dtype=np.float32)
    spatial_kappa = np.full((m_obs, args.k_max, 1, 1), np.nan, dtype=np.float32)
    spatial_u = np.full((m_obs, args.k_max, 3, 3), np.nan, dtype=np.float32)
    spatial_n = np.full((m_obs, args.k_max, 1, 1), np.nan, dtype=np.float32)
    delta_mean = np.full((m_obs, args.k_max, 3, 1), np.nan, dtype=np.float32)
    delta_kappa = np.full((m_obs, args.k_max, 1, 1), np.nan, dtype=np.float32)
    delta_u = np.full((m_obs, args.k_max, 3, 3), np.nan, dtype=np.float32)
    delta_n = np.full((m_obs, args.k_max, 1, 1), np.nan, dtype=np.float32)

    observed_point_counts = point_counts[observed_anchor_ids]
    resume_path = None
    if args.resume and posterior_path.exists():
        resume_path = posterior_path
    elif args.resume and checkpoint_path.exists():
        resume_path = checkpoint_path

    if resume_path is not None:
        completed_count = load_checkpoint(
            resume_path,
            observed_anchor_ids=observed_anchor_ids,
            final_k=final_k,
            final_elbo=final_elbo,
            selected_gain=selected_gain,
            under_modeled=under_modeled,
            fit_batch_size=fit_batch_size,
            k_growth_attempted=k_growth_attempted,
            fit_completed=fit_completed,
            alpha=alpha,
            spatial_mean=spatial_mean,
            spatial_kappa=spatial_kappa,
            spatial_u=spatial_u,
            spatial_n=spatial_n,
            delta_mean=delta_mean,
            delta_kappa=delta_kappa,
            delta_u=delta_u,
            delta_n=delta_n,
        )
        print(f"Resumed {completed_count:,} completed anchors from {resume_path}")
    elif args.resume:
        print(f"No checkpoint found at {checkpoint_path}; starting from scratch")

    fit_start = time.time()
    run_completed_count = 0

    def current_posterior_arrays() -> Dict[str, np.ndarray]:
        return posterior_arrays(
            is_observed=is_observed,
            observed_anchor_ids=observed_anchor_ids,
            point_counts=point_counts,
            final_k=final_k,
            final_elbo=final_elbo,
            selected_gain=selected_gain,
            under_modeled=under_modeled,
            fit_batch_size=fit_batch_size,
            k_growth_attempted=k_growth_attempted,
            fit_completed=fit_completed,
            alpha=alpha,
            spatial_mean=spatial_mean,
            spatial_kappa=spatial_kappa,
            spatial_u=spatial_u,
            spatial_n=spatial_n,
            delta_mean=delta_mean,
            delta_kappa=delta_kappa,
            delta_u=delta_u,
            delta_n=delta_n,
            args=args,
        )

    def current_metadata(checkpoint: bool) -> Dict:
        return build_metadata(
            args,
            device_name=device_name,
            points_norm_path=points_norm_path,
            pts_by_anchor_path=pts_by_anchor_path,
            n_anchors=n_anchors,
            m_obs=m_obs,
            total_observed=total_observed,
            observed_point_counts=observed_point_counts,
            k_growth_attempted=k_growth_attempted,
            fit_batch_size=fit_batch_size,
            under_modeled=under_modeled,
            fit_completed=fit_completed,
            adaptive_batch_size=adaptive_batch_size,
            batch_buckets=batch_buckets,
            elapsed=time.time() - fit_start,
            run_completed_count=run_completed_count,
            checkpoint_path=checkpoint_path if checkpoint else None,
        )

    def write_checkpoint() -> None:
        save_npz_atomic(checkpoint_path, **current_posterior_arrays())
        save_json(checkpoint_metadata_path, current_metadata(checkpoint=True))
        print(
            f"Wrote checkpoint {checkpoint_path} "
            f"({int(fit_completed.sum()):,}/{m_obs:,} anchors complete)"
        )

    for obs_idx, anchor_id in enumerate(observed_anchor_ids):
        if fit_completed[obs_idx]:
            continue

        anchor_point_indices = unpack_group_slice(anchor_offsets, point_indices, int(anchor_id))
        anchor_points = points_norm[anchor_point_indices]
        anchor_batch_size = choose_batch_size(
            anchor_points.shape[0],
            args.batch_size,
            batch_buckets,
            adaptive_batch_size,
        )

        cur_k = int(args.k_init)
        best_model, best_elbo = fit_anchor(
            anchor_points,
            cur_k,
            seed=args.seed + int(anchor_id),
            batch_size=anchor_batch_size,
            random_mean_init=random_mean_init,
            fit_gmm_step=fit_gmm_step,
            compute_elbo_delta=compute_elbo_delta,
            get_volume_delta_mixture=get_volume_delta_mixture,
        )

        accepted_gain = np.nan
        hit_cap_with_gain = False
        should_grow_k = anchor_points.shape[0] >= args.k_growth_min_points
        k_growth_attempted[obs_idx] = bool(should_grow_k and cur_k < args.k_max)
        while should_grow_k and cur_k < args.k_max:
            next_k = min(args.k_max, cur_k * args.k_growth_factor)
            if next_k == cur_k:
                break
            next_model, next_elbo = fit_anchor(
                anchor_points,
                next_k,
                seed=args.seed + int(anchor_id) + next_k,
                batch_size=anchor_batch_size,
                random_mean_init=random_mean_init,
                fit_gmm_step=fit_gmm_step,
                compute_elbo_delta=compute_elbo_delta,
                get_volume_delta_mixture=get_volume_delta_mixture,
            )
            gain = float(next_elbo - best_elbo)
            if gain < args.elbo_improvement_tol:
                accepted_gain = gain
                break
            best_model = next_model
            best_elbo = next_elbo
            cur_k = next_k
            accepted_gain = gain
            hit_cap_with_gain = cur_k == args.k_max and gain >= args.elbo_improvement_tol

        packed = pack_model_params(best_model, args.k_max)
        final_k[obs_idx] = packed["k"]
        final_elbo[obs_idx] = best_elbo
        selected_gain[obs_idx] = accepted_gain
        under_modeled[obs_idx] = hit_cap_with_gain
        fit_batch_size[obs_idx] = anchor_batch_size

        alpha[obs_idx] = packed["alpha"]
        spatial_mean[obs_idx] = packed["spatial_mean"]
        spatial_kappa[obs_idx] = packed["spatial_kappa"]
        spatial_u[obs_idx] = packed["spatial_u"]
        spatial_n[obs_idx] = packed["spatial_n"]
        delta_mean[obs_idx] = packed["delta_mean"]
        delta_kappa[obs_idx] = packed["delta_kappa"]
        delta_u[obs_idx] = packed["delta_u"]
        delta_n[obs_idx] = packed["delta_n"]
        fit_completed[obs_idx] = True
        run_completed_count += 1

        if (obs_idx + 1) % max(args.log_every, 1) == 0 or (obs_idx + 1) == m_obs:
            elapsed = time.time() - fit_start
            completed_count = int(fit_completed.sum())
            rate = run_completed_count / max(elapsed, 1e-6)
            print(
                f"[{obs_idx + 1:>6}/{m_obs}] anchor={anchor_id:>7} level={int(anchor_level[anchor_id])} "
                f"done={completed_count:>6} "
                f"pts={anchor_points.shape[0]:>6} K={int(final_k[obs_idx]):>2} "
                f"batch={anchor_batch_size:>4} "
                f"ELBO/pt={final_elbo[obs_idx]: .4f} rate={rate: .2f} anchors/s"
            )

        if args.checkpoint_every > 0 and run_completed_count % args.checkpoint_every == 0:
            write_checkpoint()

    if int(fit_completed.sum()) < m_obs:
        write_checkpoint()
        raise RuntimeError(
            f"Fit incomplete: {int(fit_completed.sum()):,}/{m_obs:,} anchors complete. "
            f"Resume with --resume after checking {checkpoint_path}."
        )

    save_npz_atomic(posterior_path, **current_posterior_arrays())
    metadata = current_metadata(checkpoint=False)
    save_json(metadata_path, metadata)

    print(f"Wrote {posterior_path}")
    print(f"Wrote {metadata_path}")
    print(
        f"Observed anchors: {m_obs:,} / {n_anchors:,} | "
        f"throughput={metadata['anchors_per_sec']:.2f} anchors/s | "
        f"under-modeled={metadata['under_modeled_count']:,}"
    )


if __name__ == "__main__":
    main()
