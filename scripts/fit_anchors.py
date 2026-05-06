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

from vbogs.io import cap_group_counts, save_json, select_group_values


DEFAULT_BATCH_BUCKETS = (
    "64,128,256,512,1024,2048,4096,8192,10000,16384,"
    "32768,65536,131072,262144,524288"
)
SAMPLING_METHOD = "deterministic_random_without_replacement"


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
        help="VBEM sufficient-stat batch size.",
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
        "--fit-mode",
        choices=("batched", "loop"),
        default="batched",
        help=(
            "`batched` groups anchors by point-count bucket and K; `loop` uses "
            "the original one-anchor-at-a-time implementation."
        ),
    )
    parser.add_argument(
        "--batch-buckets",
        default=DEFAULT_BATCH_BUCKETS,
        help=(
            "Comma-separated point-count buckets for `--fit-mode batched`. "
            "Each anchor is padded to the smallest bucket that fits its point count."
        ),
    )
    parser.add_argument(
        "--vmap-group-size",
        type=int,
        default=64,
        help="Maximum anchors fit together in one batched group.",
    )
    parser.add_argument(
        "--k-growth-min-points",
        type=int,
        default=256,
        help=(
            "Only attempt K growth for anchors with at least this many points. "
            "Set to 0 to attempt growth for every observed anchor."
        ),
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
        "--max-points-per-anchor",
        type=int,
        default=0,
        help=(
            "Optional cap on points used to fit each observed anchor. Dense anchors "
            "are deterministically subsampled; 0 uses every assigned point."
        ),
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


def parse_batch_buckets(raw: str, batch_size: int) -> Tuple[int, ...]:
    buckets = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not buckets:
        raise ValueError("--batch-buckets must contain at least one positive integer")
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError("--batch-buckets values must be positive")
    if buckets[-1] < batch_size:
        buckets.append(batch_size)
    return tuple(buckets)


def bucket_for_count(count: int, buckets: Tuple[int, ...]) -> int:
    for bucket in buckets:
        if count <= bucket:
            return bucket
    return count


def select_anchor_point_indices(
    anchor_offsets: np.ndarray,
    point_indices: np.ndarray,
    anchor_id: int,
    *,
    max_points_per_anchor: int,
    seed: int,
) -> np.ndarray:
    return select_group_values(
        anchor_offsets,
        point_indices,
        int(anchor_id),
        max_values_per_group=max_points_per_anchor,
        seed=seed,
    )


def cap_point_counts(point_counts: np.ndarray, max_points_per_anchor: int) -> np.ndarray:
    return cap_group_counts(point_counts, max_points_per_anchor)


def compute_mean_elbo(model, points_norm: np.ndarray, batch_size: int, compute_elbo_delta) -> float:
    total = 0.0
    count = 0
    for start in range(0, points_norm.shape[0], batch_size):
        batch = points_norm[start : start + batch_size]
        batch_jax = jnp.expand_dims(jnp.asarray(batch), axis=-1)
        elbo_arr, _ = compute_elbo_delta(model, batch_jax)
        total += float(np.asarray(elbo_arr).sum())
        count += int(batch.shape[0])
    return total / max(count, 1)


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


def make_batched_volume_delta_mixture(
    *,
    key,
    n_components: int,
    mean_init,
    n_anchors: int,
    MultivariateNormal,
    Multinomial,
    Mixture,
    DeltaMixture,
    ArrayDict,
    position_event_shape=(3, 1),
    color_event_shape=(3, 1),
    beta=0,
    learning_rate=1,
    dof_offset=1,
    position_scale=None,
    default_event_dim=2,
):
    """Batched variant of `vbgs/scripts/model_volume.py:get_volume_delta_mixture`.

    The upstream factory builds one model with batch shape `(K,)`. Here we add
    an anchor batch dimension, producing posterior tensors shaped
    `(n_anchors, K, ...)` while preserving the same prior/initial settings.
    """

    component_shape = (n_components,)
    anchor_component_shape = (n_anchors, n_components)
    if position_scale is None:
        position_scale = jnp.sqrt(n_components)

    likelihood_prior_params = MultivariateNormal.init_default_params(
        anchor_component_shape,
        position_event_shape,
        position_scale * 15,
        dof_offset=dof_offset,
        default_event_dim=default_event_dim,
    )
    likelihood_prior_params = ArrayDict(
        mean=likelihood_prior_params.mean,
        kappa=likelihood_prior_params.kappa / 1e3,
        u=likelihood_prior_params.u * 100,
        n=likelihood_prior_params.n,
    )
    likelihood_params = ArrayDict(
        mean=mean_init[:, :, :-3, :],
        kappa=likelihood_prior_params.kappa / 1e3,
        u=likelihood_prior_params.u,
        n=likelihood_prior_params.n,
    )

    delta_prior_params = MultivariateNormal.init_default_params(
        anchor_component_shape,
        color_event_shape,
        scale=1e5,
        dof_offset=dof_offset,
        default_event_dim=default_event_dim,
    )
    delta_prior_params = ArrayDict(
        mean=delta_prior_params.mean,
        kappa=delta_prior_params.kappa / 1e2,
        u=delta_prior_params.u / 100,
        n=delta_prior_params.n,
    )
    delta_params = ArrayDict(
        mean=mean_init[:, :, -3:, :],
        kappa=delta_prior_params.kappa,
        u=delta_prior_params.u * 1e5,
        n=delta_prior_params.n,
    )

    key, subkey = jr.split(key)
    prior = Multinomial(
        batch_shape=(n_anchors,),
        event_shape=component_shape,
        initial_count=1 / component_shape[0],
        init_key=subkey,
    )

    key, subkey = jr.split(key)
    likelihood = MultivariateNormal(
        batch_shape=anchor_component_shape,
        event_shape=position_event_shape,
        event_dim=len(position_event_shape),
        dof_offset=dof_offset,
        init_key=subkey,
        params=likelihood_params,
        prior_params=likelihood_prior_params,
    )

    key, subkey = jr.split(key)
    delta = MultivariateNormal(
        batch_shape=anchor_component_shape,
        event_shape=color_event_shape,
        event_dim=len(color_event_shape),
        dof_offset=dof_offset,
        init_key=subkey,
        params=delta_params,
        prior_params=delta_prior_params,
        fixed_precision=True,
    )

    opts = {"lr": learning_rate, "beta": beta}
    mixture = Mixture(likelihood, prior, pi_opts=opts, likelihood_opts=opts)
    return DeltaMixture(mixture, delta)


def fit_gmm_step_masked(initial_model, model, data_padded: np.ndarray, valid_mask: np.ndarray):
    """One VBGS sufficient-stat update for a padded anchor batch.

    `data_padded` has shape `(A, B, 6)` and `valid_mask` has shape `(A, B)`.
    Internally the sample dimension is transposed to the front so VBGS sums over
    points while retaining the anchor batch dimension.
    """

    data = jnp.expand_dims(jnp.asarray(np.swapaxes(data_padded, 0, 1)), -1)
    valid = jnp.asarray(np.swapaxes(valid_mask, 0, 1), dtype=data.dtype)

    d = initial_model.mixture.expand_to_categorical_dims(data)
    ds, dc = d[:, :, :, :-3], d[:, :, :, -3:]
    space_logprob = initial_model.mixture.likelihood.expected_log_likelihood(ds)
    color_logprob = initial_model.delta.expected_log_likelihood(dc)
    prior_logprob = initial_model.mixture.prior.log_mean()
    logprob = space_logprob + color_logprob + prior_logprob
    mixdims = tuple(range(-initial_model.mixture.prior.event_dim, 0))
    posteriors = jax.nn.softmax(logprob, mixdims) * valid[:, :, None]

    ps = initial_model.mixture._to_stats(
        posteriors,
        initial_model.mixture.get_sample_dims(d),
    )

    from vbgs.model.train import get_likelihood_sst

    ss, _ = get_likelihood_sst(
        initial_model.mixture.likelihood,
        d[:, :, :, :-3],
        posteriors,
    )
    cs, _ = get_likelihood_sst(
        initial_model.delta,
        d[:, :, :, -3:],
        posteriors,
    )

    model.mixture.prior.update_from_statistics(ps, **initial_model.mixture.pi_opts)
    model.mixture.likelihood.update_from_statistics(
        ss,
        **initial_model.mixture.likelihood_opts,
    )
    model.delta.update_from_statistics(
        cs,
        **initial_model.mixture.likelihood_opts,
    )
    return model


def compute_masked_mean_elbo(model, data_padded: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    data = jnp.expand_dims(jnp.asarray(np.swapaxes(data_padded, 0, 1)), -1)
    valid = jnp.asarray(np.swapaxes(valid_mask, 0, 1), dtype=data.dtype)

    d = model.mixture.expand_to_categorical_dims(data)
    ds, dc = d[:, :, :, :-3], d[:, :, :, -3:]
    space_logprob = model.mixture.likelihood.expected_log_likelihood(ds)
    color_logprob = model.delta.expected_log_likelihood(dc)
    prior_logprob = model.mixture.prior.log_mean()
    logprob = space_logprob + color_logprob + prior_logprob
    mixdims = tuple(range(-model.mixture.prior.event_dim, 0))
    elbo_contrib = jax.scipy.special.logsumexp(logprob, mixdims)

    prior_kl = model.mixture.prior.kl_divergence()
    space_kl = model.mixture.likelihood.kl_divergence().sum(mixdims)
    color_kl = model.delta.kl_divergence().sum(mixdims)
    elbo = elbo_contrib - space_kl - prior_kl - color_kl

    denom = jnp.maximum(valid.sum(axis=0), 1.0)
    mean_elbo = (elbo * valid).sum(axis=0) / denom
    return np.asarray(mean_elbo, dtype=np.float32)


def init_batched_means(
    anchor_ids: np.ndarray,
    data_padded: np.ndarray,
    valid_counts: np.ndarray,
    n_components: int,
    seed: int,
) -> np.ndarray:
    means = np.zeros((anchor_ids.shape[0], n_components, data_padded.shape[2], 1), dtype=np.float32)
    for row, anchor_id in enumerate(anchor_ids.tolist()):
        rng = np.random.default_rng(seed + int(anchor_id) + n_components)
        chosen = rng.integers(0, int(valid_counts[row]), size=n_components)
        means[row, :, :, 0] = data_padded[row, chosen]
        means[row, :, :, 0] += rng.normal(0.0, 0.025, size=(n_components, data_padded.shape[2])).astype(np.float32)
    return means


def pack_batched_model_params(model, k_max: int) -> Dict[str, np.ndarray]:
    n_anchors = int(np.asarray(model.mixture.prior.alpha).shape[0])
    k = int(np.asarray(model.mixture.prior.alpha).shape[1])

    alpha = np.full((n_anchors, k_max), np.nan, dtype=np.float32)
    spatial_mean = np.full((n_anchors, k_max, 3, 1), np.nan, dtype=np.float32)
    spatial_kappa = np.full((n_anchors, k_max, 1, 1), np.nan, dtype=np.float32)
    spatial_u = np.full((n_anchors, k_max, 3, 3), np.nan, dtype=np.float32)
    spatial_n = np.full((n_anchors, k_max, 1, 1), np.nan, dtype=np.float32)
    delta_mean = np.full((n_anchors, k_max, 3, 1), np.nan, dtype=np.float32)
    delta_kappa = np.full((n_anchors, k_max, 1, 1), np.nan, dtype=np.float32)
    delta_u = np.full((n_anchors, k_max, 3, 3), np.nan, dtype=np.float32)
    delta_n = np.full((n_anchors, k_max, 1, 1), np.nan, dtype=np.float32)

    alpha[:, :k] = np.asarray(model.mixture.prior.alpha, dtype=np.float32)
    spatial_mean[:, :k] = np.asarray(model.mixture.likelihood.mean, dtype=np.float32)
    spatial_kappa[:, :k] = np.asarray(model.mixture.likelihood.kappa, dtype=np.float32)
    spatial_u[:, :k] = np.asarray(model.mixture.likelihood.u, dtype=np.float32)
    spatial_n[:, :k] = np.asarray(model.mixture.likelihood.n, dtype=np.float32)
    delta_mean[:, :k] = np.asarray(model.delta.mean, dtype=np.float32)
    delta_kappa[:, :k] = np.asarray(model.delta.kappa, dtype=np.float32)
    delta_u[:, :k] = np.asarray(model.delta.u, dtype=np.float32)
    delta_n[:, :k] = np.asarray(model.delta.n, dtype=np.float32)

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


def build_padded_anchor_batch(
    *,
    anchor_ids: np.ndarray,
    bucket_size: int,
    anchor_offsets: np.ndarray,
    point_indices: np.ndarray,
    points_norm: np.ndarray,
    max_points_per_anchor: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_padded = np.zeros((anchor_ids.shape[0], bucket_size, points_norm.shape[1]), dtype=np.float32)
    valid_mask = np.zeros((anchor_ids.shape[0], bucket_size), dtype=bool)
    valid_counts = np.zeros((anchor_ids.shape[0],), dtype=np.int32)

    for row, anchor_id in enumerate(anchor_ids.tolist()):
        anchor_point_indices = select_anchor_point_indices(
            anchor_offsets,
            point_indices,
            int(anchor_id),
            max_points_per_anchor=max_points_per_anchor,
            seed=seed,
        )
        count = min(anchor_point_indices.shape[0], bucket_size)
        if count <= 0:
            continue
        data_padded[row, :count] = points_norm[anchor_point_indices[:count]]
        valid_mask[row, :count] = True
        valid_counts[row] = count

    return data_padded, valid_mask, valid_counts


def fit_batched_anchor_group(
    *,
    anchor_ids: np.ndarray,
    bucket_size: int,
    n_components: int,
    seed: int,
    anchor_offsets: np.ndarray,
    point_indices: np.ndarray,
    points_norm: np.ndarray,
    max_points_per_anchor: int,
    k_max: int,
    MultivariateNormal,
    Multinomial,
    Mixture,
    DeltaMixture,
    ArrayDict,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    data_padded, valid_mask, valid_counts = build_padded_anchor_batch(
        anchor_ids=anchor_ids,
        bucket_size=bucket_size,
        anchor_offsets=anchor_offsets,
        point_indices=point_indices,
        points_norm=points_norm,
        max_points_per_anchor=max_points_per_anchor,
        seed=seed,
    )
    mean_init = init_batched_means(
        anchor_ids=anchor_ids,
        data_padded=data_padded,
        valid_counts=valid_counts,
        n_components=n_components,
        seed=seed,
    )
    key = jr.PRNGKey(seed + int(n_components) * 100_003 + int(anchor_ids[0]))
    prior_model = make_batched_volume_delta_mixture(
        key=key,
        n_components=n_components,
        mean_init=jnp.asarray(mean_init),
        n_anchors=anchor_ids.shape[0],
        MultivariateNormal=MultivariateNormal,
        Multinomial=Multinomial,
        Mixture=Mixture,
        DeltaMixture=DeltaMixture,
        ArrayDict=ArrayDict,
        position_event_shape=(3, 1),
        beta=0,
        learning_rate=1,
    )
    model = copy.deepcopy(prior_model)
    model = fit_gmm_step_masked(prior_model, model, data_padded, valid_mask)
    elbo = compute_masked_mean_elbo(model, data_padded, valid_mask)
    packed = pack_batched_model_params(model, k_max)
    return packed, elbo, valid_counts


def assign_batched_rows(
    rows: np.ndarray,
    packed: Dict[str, np.ndarray],
    elbo: np.ndarray,
    *,
    final_k: np.ndarray,
    final_elbo: np.ndarray,
    alpha: np.ndarray,
    spatial_mean: np.ndarray,
    spatial_kappa: np.ndarray,
    spatial_u: np.ndarray,
    spatial_n: np.ndarray,
    delta_mean: np.ndarray,
    delta_kappa: np.ndarray,
    delta_u: np.ndarray,
    delta_n: np.ndarray,
) -> None:
    final_k[rows] = packed["k"]
    final_elbo[rows] = elbo
    alpha[rows] = packed["alpha"]
    spatial_mean[rows] = packed["spatial_mean"]
    spatial_kappa[rows] = packed["spatial_kappa"]
    spatial_u[rows] = packed["spatial_u"]
    spatial_n[rows] = packed["spatial_n"]
    delta_mean[rows] = packed["delta_mean"]
    delta_kappa[rows] = packed["delta_kappa"]
    delta_u[rows] = packed["delta_u"]
    delta_n[rows] = packed["delta_n"]


def main() -> None:
    args = parse_args()
    bucket_root = resolve_bucket_root(args).resolve()
    output_root = resolve_output_root(args, bucket_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    points_norm_path = (args.points_norm or (bucket_root / "points_norm.npz")).resolve()
    pts_by_anchor_path = (args.pts_by_anchor or (bucket_root / "pts_by_anchor.npz")).resolve()
    posterior_path, metadata_path = resolve_output_paths(output_root, args.max_observed_anchors)

    add_vbgs_to_path(args.vbgs_root)
    from vbgs.model.train import compute_elbo_delta, fit_gmm_step
    from vbgs.model.utils import random_mean_init
    from vbgs.model.model import DeltaMixture
    from vbgs.vi.conjugate.mvn import MultivariateNormal
    from vbgs.vi.conjugate.multinomial import Multinomial
    from vbgs.vi.models.mixture import Mixture
    from vbgs.vi.utils import ArrayDict
    from model_volume import get_volume_delta_mixture

    device_name = select_device(args.device)
    batch_buckets = parse_batch_buckets(args.batch_buckets, args.batch_size)
    if args.vmap_group_size <= 0:
        raise ValueError("--vmap-group-size must be positive")
    if args.max_points_per_anchor < 0:
        raise ValueError("--max-points-per-anchor must be non-negative")
    if 0 < args.max_points_per_anchor < args.min_points_per_anchor:
        raise ValueError(
            "--max-points-per-anchor must be 0 or at least --min-points-per-anchor"
        )

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
    fit_batch_size = np.zeros((m_obs,), dtype=np.int32)
    k_growth_attempted = np.zeros((m_obs,), dtype=bool)

    raw_observed_point_counts = point_counts[observed_anchor_ids]
    effective_point_counts = cap_point_counts(
        raw_observed_point_counts,
        args.max_points_per_anchor,
    )
    capped_anchor_count = int(np.count_nonzero(effective_point_counts < raw_observed_point_counts))
    effective_point_total = int(effective_point_counts.sum()) if m_obs else 0
    fit_point_count_processed = 0
    print(
        "Observed point counts (raw): "
        f"min={int(raw_observed_point_counts.min()) if m_obs else 0} "
        f"p50={float(np.percentile(raw_observed_point_counts, 50)) if m_obs else 0.0:.1f} "
        f"p90={float(np.percentile(raw_observed_point_counts, 90)) if m_obs else 0.0:.1f} "
        f"max={int(raw_observed_point_counts.max()) if m_obs else 0}"
    )
    print(
        "Observed point counts (fit): "
        f"min={int(effective_point_counts.min()) if m_obs else 0} "
        f"p50={float(np.percentile(effective_point_counts, 50)) if m_obs else 0.0:.1f} "
        f"p90={float(np.percentile(effective_point_counts, 90)) if m_obs else 0.0:.1f} "
        f"max={int(effective_point_counts.max()) if m_obs else 0} "
        f"capped={capped_anchor_count:,} cap={args.max_points_per_anchor}"
    )
    fit_start = time.time()

    if args.fit_mode == "loop":
        for obs_idx, anchor_id in enumerate(observed_anchor_ids):
            anchor_point_indices = select_anchor_point_indices(
                anchor_offsets,
                point_indices,
                int(anchor_id),
                max_points_per_anchor=args.max_points_per_anchor,
                seed=args.seed,
            )
            anchor_points = points_norm[anchor_point_indices]
            fit_batch_size[obs_idx] = int(anchor_points.shape[0])

            cur_k = int(args.k_init)
            best_model, best_elbo = fit_anchor(
                anchor_points,
                cur_k,
                seed=args.seed + int(anchor_id),
                batch_size=args.batch_size,
                random_mean_init=random_mean_init,
                fit_gmm_step=fit_gmm_step,
                compute_elbo_delta=compute_elbo_delta,
                get_volume_delta_mixture=get_volume_delta_mixture,
            )

            accepted_gain = np.nan
            hit_cap_with_gain = False
            while cur_k < args.k_max:
                if args.k_growth_min_points > 0 and anchor_points.shape[0] < args.k_growth_min_points:
                    break
                k_growth_attempted[obs_idx] = True
                next_k = min(args.k_max, cur_k * args.k_growth_factor)
                if next_k == cur_k:
                    break
                next_model, next_elbo = fit_anchor(
                    anchor_points,
                    next_k,
                    seed=args.seed + int(anchor_id) + next_k,
                    batch_size=args.batch_size,
                    random_mean_init=random_mean_init,
                    fit_gmm_step=fit_gmm_step,
                    compute_elbo_delta=compute_elbo_delta,
                    get_volume_delta_mixture=get_volume_delta_mixture,
                )
                gain = float(next_elbo - best_elbo)
                if gain < args.elbo_improvement_tol:
                    selected_gain[obs_idx] = gain
                    break
                best_model = next_model
                best_elbo = next_elbo
                cur_k = next_k
                accepted_gain = gain
                selected_gain[obs_idx] = accepted_gain
                hit_cap_with_gain = cur_k == args.k_max and gain >= args.elbo_improvement_tol

            packed = pack_model_params(best_model, args.k_max)
            final_k[obs_idx] = packed["k"]
            final_elbo[obs_idx] = best_elbo
            under_modeled[obs_idx] = hit_cap_with_gain

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
            fit_point_count_processed += int(anchor_points.shape[0])

            if (obs_idx + 1) % max(args.log_every, 1) == 0 or (obs_idx + 1) == m_obs:
                elapsed = time.time() - fit_start
                rate = (obs_idx + 1) / max(elapsed, 1e-6)
                print(
                    f"[{obs_idx + 1:>6}/{m_obs}] anchor={anchor_id:>7} level={int(anchor_level[anchor_id])} "
                    f"fit_pts={anchor_points.shape[0]:>6} "
                    f"fit_pts_done={fit_point_count_processed:,}/{effective_point_total:,} "
                    f"K={int(final_k[obs_idx]):>2} "
                    f"ELBO/pt={final_elbo[obs_idx]: .4f} rate={rate: .2f} anchors/s"
                )
    else:
        obs_rows = np.arange(m_obs, dtype=np.int64)
        count_to_bucket = np.array(
            [bucket_for_count(int(count), batch_buckets) for count in effective_point_counts],
            dtype=np.int32,
        )
        fit_batch_size[:] = count_to_bucket
        processed = 0

        cur_k = int(args.k_init)
        active_rows = obs_rows.copy()
        while active_rows.shape[0] > 0:
            unique_buckets = np.unique(count_to_bucket[active_rows])
            for bucket_size in unique_buckets.tolist():
                bucket_rows = active_rows[count_to_bucket[active_rows] == bucket_size]
                bucket_group_size = min(
                    args.vmap_group_size,
                    max(1, (args.vmap_group_size * args.batch_size) // max(int(bucket_size), 1)),
                )
                for start in range(0, bucket_rows.shape[0], bucket_group_size):
                    rows = bucket_rows[start : start + bucket_group_size]
                    group_anchor_ids = observed_anchor_ids[rows]
                    packed, group_elbo, _valid_counts = fit_batched_anchor_group(
                        anchor_ids=group_anchor_ids,
                        bucket_size=int(bucket_size),
                        n_components=cur_k,
                        seed=args.seed,
                        anchor_offsets=anchor_offsets,
                        point_indices=point_indices,
                        points_norm=points_norm,
                        max_points_per_anchor=args.max_points_per_anchor,
                        k_max=args.k_max,
                        MultivariateNormal=MultivariateNormal,
                        Multinomial=Multinomial,
                        Mixture=Mixture,
                        DeltaMixture=DeltaMixture,
                        ArrayDict=ArrayDict,
                    )
                    group_fit_points = int(_valid_counts.sum())

                    if cur_k == args.k_init:
                        assign_batched_rows(
                            rows,
                            packed,
                            group_elbo,
                            final_k=final_k,
                            final_elbo=final_elbo,
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
                        processed += rows.shape[0]
                        fit_point_count_processed += group_fit_points
                        fit_completed[rows] = True
                    else:
                        gain = group_elbo - final_elbo[rows]
                        selected_gain[rows] = gain
                        accept_mask = gain >= args.elbo_improvement_tol
                        if np.any(accept_mask):
                            accepted_rows = rows[accept_mask]
                            accepted_packed = {
                                key: value[accept_mask] if isinstance(value, np.ndarray) and value.shape[:1] == accept_mask.shape else value
                                for key, value in packed.items()
                            }
                            assign_batched_rows(
                                accepted_rows,
                                accepted_packed,
                                group_elbo[accept_mask],
                                final_k=final_k,
                                final_elbo=final_elbo,
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
                            if cur_k == args.k_max:
                                under_modeled[accepted_rows] = True

                    if processed and (processed % max(args.log_every, 1) == 0 or processed == m_obs):
                        elapsed = time.time() - fit_start
                        rate = processed / max(elapsed, 1e-6)
                        print(
                            f"[{processed:>6}/{m_obs}] mode=batched K={cur_k:>2} "
                            f"bucket={int(bucket_size):>6} group={rows.shape[0]:>4} "
                            f"group_fit_pts={group_fit_points:,} "
                            f"fit_pts_done={fit_point_count_processed:,}/{effective_point_total:,} "
                            f"rate={rate: .2f} anchors/s"
                        )

            next_k = min(args.k_max, cur_k * args.k_growth_factor)
            if next_k == cur_k or cur_k >= args.k_max:
                break

            growth_mask = (
                (final_k == cur_k)
                & (effective_point_counts >= max(args.k_growth_min_points, 0))
            )
            if cur_k > args.k_init:
                growth_mask &= selected_gain >= args.elbo_improvement_tol
            rows_for_growth = obs_rows[growth_mask]
            if rows_for_growth.shape[0] == 0:
                break
            k_growth_attempted[rows_for_growth] = True
            active_rows = rows_for_growth
            cur_k = next_k

    np.savez_compressed(
        posterior_path,
        is_observed=is_observed,
        observed_anchor_ids=observed_anchor_ids,
        point_count=point_counts,
        effective_point_count=effective_point_counts,
        final_k=final_k,
        final_elbo=final_elbo,
        selected_gain=selected_gain,
        under_modeled=under_modeled,
        fit_completed=fit_completed,
        fit_batch_size=fit_batch_size,
        k_growth_attempted=k_growth_attempted,
        alpha=alpha,
        spatial_mean=spatial_mean,
        spatial_kappa=spatial_kappa,
        spatial_u=spatial_u,
        spatial_n=spatial_n,
        delta_mean=delta_mean,
        delta_kappa=delta_kappa,
        delta_u=delta_u,
        delta_n=delta_n,
        k_init=np.array(args.k_init, dtype=np.int16),
        k_max=np.array(args.k_max, dtype=np.int16),
        min_points_per_anchor=np.array(args.min_points_per_anchor, dtype=np.int32),
        max_points_per_anchor=np.array(args.max_points_per_anchor, dtype=np.int32),
        elbo_improvement_tol=np.array(args.elbo_improvement_tol, dtype=np.float32),
    )

    elapsed = time.time() - fit_start
    metadata = {
        "drive": args.drive,
        "jax_device": device_name,
        "points_norm_path": str(points_norm_path),
        "pts_by_anchor_path": str(pts_by_anchor_path),
        "anchor_count": n_anchors,
        "observed_anchor_count": int(m_obs),
        "completed_anchor_count": int(fit_completed.sum()),
        "unobserved_anchor_count": int(n_anchors - int(is_observed.sum())),
        "observed_anchor_count_total": total_observed,
        "observed_point_count_min": int(raw_observed_point_counts.min()) if m_obs else 0,
        "observed_point_count_p50": float(np.percentile(raw_observed_point_counts, 50)) if m_obs else 0.0,
        "observed_point_count_p90": float(np.percentile(raw_observed_point_counts, 90)) if m_obs else 0.0,
        "observed_point_count_max": int(raw_observed_point_counts.max()) if m_obs else 0,
        "effective_point_count_min": int(effective_point_counts.min()) if m_obs else 0,
        "effective_point_count_p50": float(np.percentile(effective_point_counts, 50)) if m_obs else 0.0,
        "effective_point_count_p90": float(np.percentile(effective_point_counts, 90)) if m_obs else 0.0,
        "effective_point_count_max": int(effective_point_counts.max()) if m_obs else 0,
        "effective_point_count_total": effective_point_total,
        "max_points_per_anchor": args.max_points_per_anchor,
        "sampling_method": SAMPLING_METHOD if args.max_points_per_anchor > 0 else "none",
        "capped_anchor_count": capped_anchor_count,
        "k_init": args.k_init,
        "k_max": args.k_max,
        "k_growth_factor": args.k_growth_factor,
        "k_growth_min_points": args.k_growth_min_points,
        "batch_size": args.batch_size,
        "batch_buckets": list(batch_buckets),
        "vmap_group_size": args.vmap_group_size,
        "fit_mode": args.fit_mode,
        "min_points_per_anchor": args.min_points_per_anchor,
        "elbo_improvement_tol": args.elbo_improvement_tol,
        "elapsed_sec": elapsed,
        "anchors_per_sec": m_obs / max(elapsed, 1e-6),
        "under_modeled_count": int(under_modeled.sum()),
        "k_growth_attempted_count": int(k_growth_attempted.sum()),
        "fit_batch_size_min": int(fit_batch_size.min()) if m_obs else 0,
        "fit_batch_size_p50": float(np.percentile(fit_batch_size, 50)) if m_obs else 0.0,
        "fit_batch_size_p90": float(np.percentile(fit_batch_size, 90)) if m_obs else 0.0,
        "fit_batch_size_max": int(fit_batch_size.max()) if m_obs else 0,
        "max_observed_anchors": args.max_observed_anchors,
    }
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
