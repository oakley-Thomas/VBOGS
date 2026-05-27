"""Online posterior-state utilities for the VBOGS real-time loop."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ONLINE_STATE_VERSION = 2
ROW_FIELDS = (
    "alpha",
    "spatial_mean",
    "spatial_kappa",
    "spatial_u",
    "spatial_n",
    "delta_mean",
    "delta_kappa",
    "delta_u",
    "delta_n",
)
INITIAL_ROW_FIELDS = (
    "initial_alpha",
    "initial_spatial_mean",
    "initial_spatial_kappa",
    "initial_spatial_u",
    "initial_spatial_n",
    "initial_delta_mean",
    "initial_delta_kappa",
    "initial_delta_u",
    "initial_delta_n",
)


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        tmp_name = handle.name
        np.savez_compressed(handle, **arrays)
    os.replace(tmp_name, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        tmp_name = handle.name
        np.save(handle, array)
    os.replace(tmp_name, path)


def expand_posterior_to_anchor_rows(posterior: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Expand offline observed-row posterior arrays to one row per anchor.

    Offline `anchor_posterior.npz` stores parameters only for anchors that were
    sufficiently observed. Online updates need addressable rows for every fixed
    Octree-AnyGS anchor, including anchors that become observed later.
    """

    is_observed = np.asarray(posterior["is_observed"], dtype=bool).reshape(-1)
    observed_anchor_ids = np.asarray(posterior["observed_anchor_ids"], dtype=np.int64).reshape(-1)
    anchor_count = int(is_observed.shape[0])
    if observed_anchor_ids.size:
        k_max = int(np.asarray(posterior["alpha"]).shape[1])
    else:
        k_max = int(np.asarray(posterior["k_max"])) if "k_max" in posterior else 1

    expanded: dict[str, np.ndarray] = {
        "state_version": np.array(ONLINE_STATE_VERSION, dtype=np.int16),
        "is_observed": is_observed.copy(),
        "observed_anchor_ids": np.arange(anchor_count, dtype=np.int64),
        "point_count": np.asarray(
            posterior["point_count"] if "point_count" in posterior else np.zeros(anchor_count),
            dtype=np.int32,
        ).reshape(anchor_count),
        "final_k": np.zeros(anchor_count, dtype=np.int16),
        "final_elbo": np.full(anchor_count, np.nan, dtype=np.float32),
        "selected_gain": np.full(anchor_count, np.nan, dtype=np.float32),
        "under_modeled": np.zeros(anchor_count, dtype=bool),
        "fit_completed": np.zeros(anchor_count, dtype=bool),
        "fit_batch_size": np.zeros(anchor_count, dtype=np.int32),
        "k_growth_attempted": np.zeros(anchor_count, dtype=bool),
        "online_update_count": np.zeros(anchor_count, dtype=np.int32),
        "online_observation_count": np.zeros(anchor_count, dtype=np.int32),
        "online_sum6": np.zeros((anchor_count, 6), dtype=np.float32),
        "online_outer6": np.zeros((anchor_count, 6, 6), dtype=np.float32),
        "pending_points_norm": np.empty((0, 6), dtype=np.float32),
        "pending_anchor_ids": np.empty((0,), dtype=np.int64),
        "last_update_seq": np.array(-1, dtype=np.int64),
    }

    for name in ("k_init", "k_max", "min_points_per_anchor", "elbo_improvement_tol"):
        if name in posterior:
            expanded[name] = np.asarray(posterior[name])

    for name in ROW_FIELDS:
        source = np.asarray(posterior[name], dtype=np.float32)
        shape = (anchor_count, *source.shape[1:])
        target = np.full(shape, np.nan, dtype=np.float32)
        if observed_anchor_ids.size:
            target[observed_anchor_ids] = source
        expanded[name] = target

    for name in INITIAL_ROW_FIELDS:
        if name in posterior:
            source = np.asarray(posterior[name], dtype=np.float32)
            shape = (anchor_count, *source.shape[1:])
            target = np.full(shape, np.nan, dtype=np.float32)
            if observed_anchor_ids.size:
                target[observed_anchor_ids] = source
            expanded[name] = target
        else:
            reference_name = name.removeprefix("initial_")
            expanded[name] = np.full_like(expanded[reference_name], np.nan, dtype=np.float32)

    for name, dtype in (
        ("final_k", np.int16),
        ("final_elbo", np.float32),
        ("selected_gain", np.float32),
        ("under_modeled", bool),
        ("fit_completed", bool),
        ("fit_batch_size", np.int32),
        ("k_growth_attempted", bool),
    ):
        if name not in posterior:
            continue
        source = np.asarray(posterior[name], dtype=dtype)
        expanded[name][observed_anchor_ids] = source

    return expanded


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1.0e-12, None)


def _safe_inverse(mat: np.ndarray, eps: float) -> np.ndarray:
    mat = 0.5 * (mat + mat.T)
    try:
        return np.linalg.inv(mat + np.eye(mat.shape[0], dtype=np.float64) * eps)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(mat + np.eye(mat.shape[0], dtype=np.float64) * eps)


def _spatial_prior_params(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Mirrors vbgs/scripts/model_volume.py:get_volume_delta_mixture.
    u = np.eye(3, dtype=np.float32) * (22500.0 * float(k))
    kappa = np.array([[1.0e-3]], dtype=np.float32)
    n = np.array([[5.0]], dtype=np.float32)
    return kappa, u, n


def _spatial_initial_params(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior_kappa, prior_u, prior_n = _spatial_prior_params(k)
    return prior_kappa / 1.0e3, prior_u, prior_n


def _delta_prior_params() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.eye(3, dtype=np.float32) * 1.0e8
    kappa = np.array([[1.0e-2]], dtype=np.float32)
    n = np.array([[5.0]], dtype=np.float32)
    return kappa, u, n


def _initial_means_from_points(points_norm: np.ndarray, anchor_id: int, k: int, seed: int) -> np.ndarray:
    if points_norm.shape[0] == 0:
        return np.zeros((k, 6, 1), dtype=np.float32)
    rng = np.random.default_rng(int(seed) + int(anchor_id) + int(k))
    chosen = rng.integers(0, int(points_norm.shape[0]), size=int(k))
    means = points_norm[chosen].astype(np.float32, copy=True)
    means += rng.normal(0.0, 0.025, size=means.shape).astype(np.float32)
    return means[:, :, None]


def _fill_initial_fields_for_anchor(
    state: dict[str, np.ndarray],
    anchor_id: int,
    initial_mean6: np.ndarray,
    *,
    eps: float,
) -> None:
    k = int(initial_mean6.shape[0])
    if k <= 0:
        return
    spatial_kappa, spatial_u, spatial_n = _spatial_initial_params(k)
    delta_kappa, delta_u, delta_n = _delta_prior_params()

    for name in INITIAL_ROW_FIELDS:
        state[name][anchor_id, :] = np.nan

    state["initial_alpha"][anchor_id, :k] = np.full((k,), 1.0 / max(k, 1), dtype=np.float32)
    state["initial_spatial_mean"][anchor_id, :k] = initial_mean6[:, :3]
    state["initial_spatial_kappa"][anchor_id, :k] = spatial_kappa
    state["initial_spatial_u"][anchor_id, :k] = spatial_u
    state["initial_spatial_n"][anchor_id, :k] = spatial_n
    state["initial_delta_mean"][anchor_id, :k] = initial_mean6[:, 3:]
    state["initial_delta_kappa"][anchor_id, :k] = delta_kappa
    state["initial_delta_u"][anchor_id, :k] = delta_u
    state["initial_delta_n"][anchor_id, :k] = delta_n

    # Avoid zero precision if a caller initialized with degenerate constants.
    state["initial_spatial_u"][anchor_id, :k] = np.maximum(
        state["initial_spatial_u"][anchor_id, :k],
        np.eye(3, dtype=np.float32) * eps,
    )
    state["initial_delta_u"][anchor_id, :k] = np.maximum(
        state["initial_delta_u"][anchor_id, :k],
        np.eye(3, dtype=np.float32) * eps,
    )


def backfill_initial_fields_from_points(
    state: Mapping[str, np.ndarray],
    *,
    points_norm: np.ndarray,
    anchor_offsets: np.ndarray,
    point_indices: np.ndarray,
    seed: int = 0,
    eps: float = 1.0e-6,
) -> dict[str, np.ndarray]:
    """Fill missing v2 VBGS initial fields from packed offline point buckets.

    New M4b posteriors save these fields directly. This compatibility path is
    deterministic and is intended for older artifacts that only saved posterior
    rows.
    """

    updated = {key: np.array(value, copy=True) for key, value in state.items()}
    final_k = np.asarray(updated["final_k"], dtype=np.int32)
    fit_completed = np.asarray(updated["fit_completed"], dtype=bool)
    missing_initial = ~np.isfinite(updated["initial_spatial_mean"]).any(axis=(1, 2, 3))
    for anchor_id in np.nonzero(fit_completed & missing_initial)[0].tolist():
        k = int(final_k[anchor_id])
        if k <= 0:
            continue
        start = int(anchor_offsets[anchor_id])
        end = int(anchor_offsets[anchor_id + 1])
        anchor_points = points_norm[point_indices[start:end]]
        initial_mean6 = _initial_means_from_points(anchor_points, anchor_id, k, seed)
        _fill_initial_fields_for_anchor(updated, anchor_id, initial_mean6, eps=eps)
    return updated


def _logpdf_precision(x: np.ndarray, mean: np.ndarray, precision: np.ndarray, eps: float) -> np.ndarray:
    precision = 0.5 * (precision + precision.T)
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0 or not np.isfinite(logdet):
        eigvals = np.linalg.eigvalsh(precision)
        logdet = float(np.sum(np.log(np.clip(eigvals, eps, None))))
        precision = np.diag(np.clip(eigvals, eps, None))
    diff = x - mean[None, :]
    quad = np.einsum("ni,ij,nj->n", diff, precision, diff)
    dim = x.shape[1]
    return 0.5 * (logdet - dim * np.log(2.0 * np.pi) - quad)


def _responsibilities_from_initial(
    state: Mapping[str, np.ndarray],
    anchor_id: int,
    points_norm: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    k = int(state["final_k"][anchor_id])
    if k <= 0:
        return np.zeros((points_norm.shape[0], 0), dtype=np.float64)
    alpha = state["initial_alpha"][anchor_id, :k].astype(np.float64)
    if not np.isfinite(alpha).all() or float(alpha.sum()) <= eps:
        alpha = np.full((k,), 1.0 / max(k, 1), dtype=np.float64)
    log_prior = np.log(np.clip(alpha, eps, None)) - np.log(np.clip(alpha.sum(), eps, None))
    spatial_points = points_norm[:, :3].astype(np.float64)
    delta_points = points_norm[:, 3:].astype(np.float64)
    logits = np.empty((points_norm.shape[0], k), dtype=np.float64)
    for comp in range(k):
        spatial_mean = state["initial_spatial_mean"][anchor_id, comp, :, 0].astype(np.float64)
        spatial_precision = (
            float(state["initial_spatial_n"][anchor_id, comp, 0, 0])
            * state["initial_spatial_u"][anchor_id, comp].astype(np.float64)
        )
        delta_mean = state["initial_delta_mean"][anchor_id, comp, :, 0].astype(np.float64)
        delta_precision = (
            float(state["initial_delta_n"][anchor_id, comp, 0, 0])
            * state["initial_delta_u"][anchor_id, comp].astype(np.float64)
        )
        logits[:, comp] = (
            log_prior[comp]
            + _logpdf_precision(spatial_points, spatial_mean, spatial_precision, eps)
            + _logpdf_precision(delta_points, delta_mean, delta_precision, eps)
        )
    return _softmax(logits)


def _canonical_to_natural(
    mean: np.ndarray,
    kappa: float,
    u: np.ndarray,
    n: float,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = mean.astype(np.float64)
    eta1 = mean * kappa
    eta2 = -0.5 * (_safe_inverse(u.astype(np.float64), eps) + kappa * np.outer(mean, mean))
    return eta1, eta2, float(kappa), float(n) - float(mean.shape[0])


def _natural_to_canonical(
    eta1: np.ndarray,
    eta2: np.ndarray,
    nu1: float,
    nu2: float,
    *,
    eps: float,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    kappa = max(float(nu1), eps)
    mean = eta1 / kappa
    inv_u = -2.0 * eta2 - (1.0 / kappa) * np.outer(eta1, eta1)
    u = _safe_inverse(inv_u, eps)
    n = float(nu2) + float(mean.shape[0])
    return mean.astype(np.float32), kappa, u.astype(np.float32), n


def _update_observed_anchor_exact_fixed_k(
    state: dict[str, np.ndarray],
    anchor_id: int,
    points_norm: np.ndarray,
    *,
    eps: float,
) -> None:
    k = int(state["final_k"][anchor_id])
    if k <= 0:
        return
    resp = _responsibilities_from_initial(state, anchor_id, points_norm, eps=eps)
    spatial_points = points_norm[:, :3].astype(np.float64)
    delta_points = points_norm[:, 3:].astype(np.float64)
    max_resp_mean = float(resp.max(axis=1).mean()) if resp.size else 1.0

    for comp in range(k):
        weights = resp[:, comp]
        nk = float(weights.sum())
        if nk <= eps:
            continue
        state["alpha"][anchor_id, comp] = np.float32(float(state["alpha"][anchor_id, comp]) + nk)

        spatial_sum = (spatial_points * weights[:, None]).sum(axis=0)
        spatial_xx = (spatial_points * weights[:, None]).T @ spatial_points
        mean = state["spatial_mean"][anchor_id, comp, :, 0].astype(np.float64)
        kappa = float(state["spatial_kappa"][anchor_id, comp, 0, 0])
        u = state["spatial_u"][anchor_id, comp].astype(np.float64)
        n = float(state["spatial_n"][anchor_id, comp, 0, 0])
        eta1, eta2, nu1, nu2 = _canonical_to_natural(mean, kappa, u, n, eps=eps)
        eta1 += spatial_sum
        eta2 += -0.5 * spatial_xx
        nu1 += nk
        nu2 += nk
        new_mean, new_kappa, new_u, new_n = _natural_to_canonical(eta1, eta2, nu1, nu2, eps=eps)
        state["spatial_mean"][anchor_id, comp, :, 0] = new_mean
        state["spatial_kappa"][anchor_id, comp, 0, 0] = np.float32(new_kappa)
        state["spatial_u"][anchor_id, comp] = new_u
        state["spatial_n"][anchor_id, comp, 0, 0] = np.float32(new_n)

        delta_sum = (delta_points * weights[:, None]).sum(axis=0)
        old_delta_kappa = float(state["delta_kappa"][anchor_id, comp, 0, 0])
        old_delta_mean = state["delta_mean"][anchor_id, comp, :, 0].astype(np.float64)
        new_delta_kappa = max(old_delta_kappa + nk, eps)
        new_delta_mean = (old_delta_kappa * old_delta_mean + delta_sum) / new_delta_kappa
        state["delta_mean"][anchor_id, comp, :, 0] = new_delta_mean.astype(np.float32)
        state["delta_kappa"][anchor_id, comp, 0, 0] = np.float32(new_delta_kappa)

    if "under_modeled" in state:
        k_max = int(np.asarray(state.get("k_max", np.array(state["alpha"].shape[1]))))
        if k >= k_max and max_resp_mean < 0.65:
            state["under_modeled"][anchor_id] = True


def _pending_points_for_anchor(state: Mapping[str, np.ndarray], anchor_id: int) -> np.ndarray:
    pending_points = np.asarray(state.get("pending_points_norm", np.empty((0, 6))), dtype=np.float32)
    pending_ids = np.asarray(state.get("pending_anchor_ids", np.empty((0,), dtype=np.int64)), dtype=np.int64)
    if pending_points.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float32)
    return pending_points[pending_ids == int(anchor_id)]


def _append_pending_points(state: dict[str, np.ndarray], anchor_id: int, points_norm: np.ndarray) -> None:
    if points_norm.shape[0] == 0:
        return
    pending_points = np.asarray(state.get("pending_points_norm", np.empty((0, 6))), dtype=np.float32)
    pending_ids = np.asarray(state.get("pending_anchor_ids", np.empty((0,), dtype=np.int64)), dtype=np.int64)
    state["pending_points_norm"] = np.concatenate([pending_points, points_norm.astype(np.float32)], axis=0)
    state["pending_anchor_ids"] = np.concatenate(
        [pending_ids, np.full((points_norm.shape[0],), int(anchor_id), dtype=np.int64)],
        axis=0,
    )


def _remove_pending_anchor(state: dict[str, np.ndarray], anchor_id: int) -> None:
    pending_points = np.asarray(state.get("pending_points_norm", np.empty((0, 6))), dtype=np.float32)
    pending_ids = np.asarray(state.get("pending_anchor_ids", np.empty((0,), dtype=np.int64)), dtype=np.int64)
    if pending_points.shape[0] == 0:
        return
    keep = pending_ids != int(anchor_id)
    state["pending_points_norm"] = pending_points[keep]
    state["pending_anchor_ids"] = pending_ids[keep]


def _initialize_exact_anchor_from_points(
    state: dict[str, np.ndarray],
    anchor_id: int,
    points_norm: np.ndarray,
    *,
    eps: float,
) -> None:
    if points_norm.shape[0] == 0:
        return
    k_max = int(state["alpha"].shape[1])
    k_init = int(np.asarray(state.get("k_init", np.array(min(k_max, 1)))))
    k = max(1, min(k_init, k_max))
    initial_mean6 = _initial_means_from_points(points_norm, anchor_id, k, seed=0)
    _fill_initial_fields_for_anchor(state, anchor_id, initial_mean6, eps=eps)

    for name in ROW_FIELDS:
        state[name][anchor_id, :] = np.nan
    spatial_kappa, spatial_u, spatial_n = _spatial_initial_params(k)
    delta_kappa, delta_u, delta_n = _delta_prior_params()
    state["alpha"][anchor_id, :k] = np.full((k,), 1.0 / max(k, 1), dtype=np.float32)
    state["spatial_mean"][anchor_id, :k] = initial_mean6[:, :3]
    state["spatial_kappa"][anchor_id, :k] = spatial_kappa
    state["spatial_u"][anchor_id, :k] = spatial_u
    state["spatial_n"][anchor_id, :k] = spatial_n
    state["delta_mean"][anchor_id, :k] = initial_mean6[:, 3:]
    state["delta_kappa"][anchor_id, :k] = delta_kappa
    state["delta_u"][anchor_id, :k] = delta_u
    state["delta_n"][anchor_id, :k] = delta_n
    state["final_k"][anchor_id] = k
    state["fit_completed"][anchor_id] = True
    state["is_observed"][anchor_id] = True
    _update_observed_anchor_exact_fixed_k(state, anchor_id, points_norm, eps=eps)


def _weighted_cov(points: np.ndarray, weights: np.ndarray, mean: np.ndarray, eps: float) -> np.ndarray:
    nk = float(weights.sum())
    if nk <= eps:
        return np.eye(points.shape[1], dtype=np.float64)
    diff = points - mean[None, :]
    cov = (diff * weights[:, None]).T @ diff / max(nk, eps)
    return 0.5 * (cov + cov.T)


def _initialize_anchor_from_points(
    state: dict[str, np.ndarray],
    anchor_id: int,
    points_norm: np.ndarray,
    *,
    eps: float,
) -> None:
    if points_norm.shape[0] == 0:
        return
    k_max = int(state["alpha"].shape[1])
    final_k = 1 if k_max > 0 else 0
    if final_k == 0:
        return
    mean6 = points_norm.mean(axis=0)
    centered = points_norm - mean6[None, :]
    cov6 = centered.T @ centered / max(points_norm.shape[0], 1)
    spatial_cov = cov6[:3, :3]
    delta_cov = cov6[3:, 3:]

    state["alpha"][anchor_id, :] = np.nan
    state["spatial_mean"][anchor_id, :] = np.nan
    state["spatial_kappa"][anchor_id, :] = np.nan
    state["spatial_u"][anchor_id, :] = np.nan
    state["spatial_n"][anchor_id, :] = np.nan
    state["delta_mean"][anchor_id, :] = np.nan
    state["delta_kappa"][anchor_id, :] = np.nan
    state["delta_u"][anchor_id, :] = np.nan
    state["delta_n"][anchor_id, :] = np.nan

    count = float(points_norm.shape[0])
    state["alpha"][anchor_id, 0] = count + 1.0
    state["spatial_mean"][anchor_id, 0, :, 0] = mean6[:3]
    state["spatial_kappa"][anchor_id, 0, 0, 0] = max(count, eps)
    state["spatial_u"][anchor_id, 0] = _safe_inverse(spatial_cov, eps).astype(np.float32)
    state["spatial_n"][anchor_id, 0, 0, 0] = max(count + 3.0, 4.0 + eps)
    state["delta_mean"][anchor_id, 0, :, 0] = mean6[3:]
    state["delta_kappa"][anchor_id, 0, 0, 0] = max(count, eps)
    state["delta_u"][anchor_id, 0] = _safe_inverse(delta_cov, eps).astype(np.float32)
    state["delta_n"][anchor_id, 0, 0, 0] = max(count + 3.0, 4.0 + eps)
    state["final_k"][anchor_id] = final_k
    state["fit_completed"][anchor_id] = True
    state["is_observed"][anchor_id] = True


def _update_observed_anchor(
    state: dict[str, np.ndarray],
    anchor_id: int,
    points_norm: np.ndarray,
    *,
    eps: float,
) -> None:
    k = int(state["final_k"][anchor_id])
    if k <= 0 or not bool(state["fit_completed"][anchor_id]):
        _initialize_anchor_from_points(state, anchor_id, points_norm, eps=eps)
        return

    spatial_points = points_norm[:, :3].astype(np.float64)
    delta_points = points_norm[:, 3:].astype(np.float64)
    alpha = np.clip(state["alpha"][anchor_id, :k].astype(np.float64), eps, None)
    log_prior = np.log(alpha) - np.log(alpha.sum())
    logits = np.empty((points_norm.shape[0], k), dtype=np.float64)
    for comp in range(k):
        spatial_mean = state["spatial_mean"][anchor_id, comp, :, 0].astype(np.float64)
        delta_mean = state["delta_mean"][anchor_id, comp, :, 0].astype(np.float64)
        spatial_precision = (
            float(state["spatial_n"][anchor_id, comp, 0, 0])
            * state["spatial_u"][anchor_id, comp].astype(np.float64)
        )
        delta_precision = (
            float(state["delta_n"][anchor_id, comp, 0, 0])
            * state["delta_u"][anchor_id, comp].astype(np.float64)
        )
        logits[:, comp] = (
            log_prior[comp]
            + _logpdf_precision(spatial_points, spatial_mean, spatial_precision, eps)
            + _logpdf_precision(delta_points, delta_mean, delta_precision, eps)
        )

    resp = _softmax(logits)
    max_resp_mean = float(resp.max(axis=1).mean()) if resp.size else 1.0
    for comp in range(k):
        weights = resp[:, comp]
        nk = float(weights.sum())
        if nk <= eps:
            continue
        old_alpha = float(state["alpha"][anchor_id, comp])
        old_mass = max(old_alpha, eps)
        new_mass = old_mass + nk

        for prefix, points in (("spatial", spatial_points), ("delta", delta_points)):
            mean_key = f"{prefix}_mean"
            kappa_key = f"{prefix}_kappa"
            u_key = f"{prefix}_u"
            n_key = f"{prefix}_n"
            old_mean = state[mean_key][anchor_id, comp, :, 0].astype(np.float64)
            new_batch_mean = (points * weights[:, None]).sum(axis=0) / max(nk, eps)
            old_cov = _safe_inverse(state[u_key][anchor_id, comp].astype(np.float64), eps)
            batch_cov = _weighted_cov(points, weights, new_batch_mean, eps)
            combined_mean = (old_mass * old_mean + nk * new_batch_mean) / new_mass
            old_shift = old_mean - combined_mean
            new_shift = new_batch_mean - combined_mean
            combined_cov = (
                old_mass * (old_cov + np.outer(old_shift, old_shift))
                + nk * (batch_cov + np.outer(new_shift, new_shift))
            ) / max(new_mass, eps)
            state[mean_key][anchor_id, comp, :, 0] = combined_mean.astype(np.float32)
            state[kappa_key][anchor_id, comp, 0, 0] = np.float32(
                float(state[kappa_key][anchor_id, comp, 0, 0]) + nk
            )
            state[n_key][anchor_id, comp, 0, 0] = np.float32(
                float(state[n_key][anchor_id, comp, 0, 0]) + nk
            )
            state[u_key][anchor_id, comp] = _safe_inverse(combined_cov, eps).astype(np.float32)

        state["alpha"][anchor_id, comp] = np.float32(old_alpha + nk)

    if "under_modeled" in state:
        k_max = int(np.asarray(state.get("k_max", np.array(state["alpha"].shape[1]))))
        if k >= k_max and max_resp_mean < 0.65:
            state["under_modeled"][anchor_id] = True


def apply_online_moment_update(
    state: Mapping[str, np.ndarray],
    batch: Mapping[str, np.ndarray],
    *,
    seq: int,
    min_points_per_anchor: int,
    eps: float = 1.0e-6,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply a deterministic fixed-K online update to touched anchors.

    This is the first online updater. It keeps the offline-selected K fixed and
    updates posterior parameters from responsibility-weighted moment statistics.
    The entry point is intentionally small so a later exact VBGS `fit_gmm_step`
    state restoration can replace the update kernel without changing ROS/file
    contracts.
    """

    updated = {key: np.array(value, copy=True) for key, value in state.items()}
    points_norm = np.asarray(batch["points_norm"], dtype=np.float32)
    anchor_offsets = np.asarray(batch["anchor_offsets"], dtype=np.int64)
    point_indices = np.asarray(batch["point_indices"], dtype=np.int64)
    touched_anchor_ids = np.asarray(batch["touched_anchor_ids"], dtype=np.int64)
    touched_updated: list[int] = []
    touched_deferred: list[int] = []

    for anchor_id in touched_anchor_ids.tolist():
        start = int(anchor_offsets[anchor_id])
        end = int(anchor_offsets[anchor_id + 1])
        if end <= start:
            continue
        anchor_points = points_norm[point_indices[start:end]]
        if anchor_points.size == 0:
            continue

        updated["online_observation_count"][anchor_id] += int(anchor_points.shape[0])
        updated["online_sum6"][anchor_id] += anchor_points.sum(axis=0)
        updated["online_outer6"][anchor_id] += np.einsum("ni,nj->ij", anchor_points, anchor_points)
        updated["point_count"][anchor_id] += int(anchor_points.shape[0])

        if bool(updated["fit_completed"][anchor_id]):
            _update_observed_anchor(updated, anchor_id, anchor_points, eps=eps)
            touched_updated.append(int(anchor_id))
        elif int(updated["online_observation_count"][anchor_id]) >= min_points_per_anchor:
            count = int(updated["online_observation_count"][anchor_id])
            mean = updated["online_sum6"][anchor_id] / max(count, 1)
            cov = updated["online_outer6"][anchor_id] / max(count, 1) - np.outer(mean, mean)
            rng = np.random.default_rng(int(anchor_id))
            synthetic = rng.multivariate_normal(
                mean.astype(np.float64),
                cov.astype(np.float64) + np.eye(6) * eps,
                size=max(count, 1),
                method="eigh",
            ).astype(np.float32)
            _initialize_anchor_from_points(updated, anchor_id, synthetic, eps=eps)
            touched_updated.append(int(anchor_id))
        else:
            touched_deferred.append(int(anchor_id))

        updated["online_update_count"][anchor_id] += 1

    updated["last_update_seq"] = np.array(seq, dtype=np.int64)
    metadata = {
        "seq": int(seq),
        "touched_anchor_count": int(touched_anchor_ids.shape[0]),
        "updated_anchor_count": int(len(touched_updated)),
        "deferred_anchor_count": int(len(touched_deferred)),
        "updated_anchor_ids": touched_updated,
        "deferred_anchor_ids": touched_deferred,
        "update_mode": "fixed_k_moment",
    }
    return updated, metadata


def apply_online_exact_fixed_k_update(
    state: Mapping[str, np.ndarray],
    batch: Mapping[str, np.ndarray],
    *,
    seq: int,
    min_points_per_anchor: int,
    eps: float = 1.0e-6,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply the v2 fixed-scaffold exact-K online VBGS update.

    The Octree anchor topology remains fixed. Existing anchors keep their
    offline-selected `final_k`; previously unobserved anchors initialize with
    `k_init` once enough pending online points have accumulated.
    """

    updated = {key: np.array(value, copy=True) for key, value in state.items()}
    if "pending_points_norm" not in updated:
        updated["pending_points_norm"] = np.empty((0, 6), dtype=np.float32)
    if "pending_anchor_ids" not in updated:
        updated["pending_anchor_ids"] = np.empty((0,), dtype=np.int64)

    points_norm = np.asarray(batch["points_norm"], dtype=np.float32)
    anchor_offsets = np.asarray(batch["anchor_offsets"], dtype=np.int64)
    point_indices = np.asarray(batch["point_indices"], dtype=np.int64)
    touched_anchor_ids = np.asarray(batch["touched_anchor_ids"], dtype=np.int64)
    touched_updated: list[int] = []
    touched_deferred: list[int] = []

    for anchor_id in touched_anchor_ids.tolist():
        start = int(anchor_offsets[anchor_id])
        end = int(anchor_offsets[anchor_id + 1])
        if end <= start:
            continue
        anchor_points = points_norm[point_indices[start:end]]
        if anchor_points.size == 0:
            continue

        updated["online_observation_count"][anchor_id] += int(anchor_points.shape[0])
        updated["online_sum6"][anchor_id] += anchor_points.sum(axis=0)
        updated["online_outer6"][anchor_id] += np.einsum("ni,nj->ij", anchor_points, anchor_points)
        updated["point_count"][anchor_id] += int(anchor_points.shape[0])

        if bool(updated["fit_completed"][anchor_id]):
            if not np.isfinite(updated["initial_spatial_mean"][anchor_id]).any():
                k = int(updated["final_k"][anchor_id])
                fallback_mean6 = np.concatenate(
                    [
                        updated["spatial_mean"][anchor_id, :k],
                        updated["delta_mean"][anchor_id, :k],
                    ],
                    axis=1,
                )
                _fill_initial_fields_for_anchor(updated, anchor_id, fallback_mean6, eps=eps)
            _update_observed_anchor_exact_fixed_k(updated, anchor_id, anchor_points, eps=eps)
            touched_updated.append(int(anchor_id))
        else:
            _append_pending_points(updated, anchor_id, anchor_points)
            pending_points = _pending_points_for_anchor(updated, anchor_id)
            if pending_points.shape[0] >= int(min_points_per_anchor):
                _initialize_exact_anchor_from_points(updated, anchor_id, pending_points, eps=eps)
                _remove_pending_anchor(updated, anchor_id)
                touched_updated.append(int(anchor_id))
            else:
                touched_deferred.append(int(anchor_id))

        updated["online_update_count"][anchor_id] += 1

    updated["last_update_seq"] = np.array(seq, dtype=np.int64)
    metadata = {
        "seq": int(seq),
        "touched_anchor_count": int(touched_anchor_ids.shape[0]),
        "updated_anchor_count": int(len(touched_updated)),
        "deferred_anchor_count": int(len(touched_deferred)),
        "updated_anchor_ids": touched_updated,
        "deferred_anchor_ids": touched_deferred,
        "update_mode": "exact_fixed_k",
        "pending_point_count": int(updated["pending_points_norm"].shape[0]),
    }
    return updated, metadata
