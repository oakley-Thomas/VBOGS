"""Online posterior-state utilities for the VBOGS real-time loop."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ONLINE_STATE_VERSION = 1
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
