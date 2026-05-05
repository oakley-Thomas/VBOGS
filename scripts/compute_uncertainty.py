#!/usr/bin/env python3

"""Reduce per-anchor VBGS posteriors to scalar uncertainty values.

This is M5 from PLAN.md. It reads the M4b `anchor_posterior.npz` artifact and
writes:

- `U.npy`: one scalar uncertainty per Octree-AnyGS anchor
- `uncertainty_components.npz`: diagnostic entropy terms
- `uncertainty_metadata.json`: provenance and summary statistics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.special import digamma, gammaln

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.io import save_json


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
        help="Directory containing M4/M5 artifacts. Defaults to `data/m4/<drive>`.",
    )
    parser.add_argument(
        "--posterior",
        type=Path,
        default=None,
        help="Explicit path to `anchor_posterior.npz`. Defaults to full, then smoke posterior.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where M5 artifacts will be written. Defaults to the posterior directory.",
    )
    parser.add_argument(
        "--output-name",
        default="U.npy",
        help="Filename for the per-anchor uncertainty vector.",
    )
    parser.add_argument(
        "--components-name",
        default="uncertainty_components.npz",
        help="Filename for diagnostic entropy components.",
    )
    parser.add_argument(
        "--metadata-name",
        default="uncertainty_metadata.json",
        help="Filename for summary metadata.",
    )
    parser.add_argument(
        "--u-max",
        type=float,
        default=None,
        help=(
            "Uncertainty assigned to unobserved/unfitted anchors. Defaults to the "
            "maximum finite fitted uncertainty, or 1.0 if no fitted anchors exist."
        ),
    )
    parser.add_argument(
        "--include-dirichlet-entropy",
        action="store_true",
        help=(
            "Add Dirichlet entropy of the mixture weights to U. Default is off to "
            "match Algorithm.txt's pi-weighted per-component entropy definition."
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Small positive floor used for numerical stability.",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=64,
        help="Number of bins used for the fitted-anchor uncertainty histogram diagnostic.",
    )
    return parser.parse_args()


def resolve_bucket_root(args: argparse.Namespace) -> Path:
    if args.bucket_root is not None:
        return args.bucket_root.resolve()
    return (Path("data/m4") / args.drive).resolve()


def resolve_posterior_path(args: argparse.Namespace, bucket_root: Path) -> Path:
    if args.posterior is not None:
        return args.posterior.resolve()
    full_path = bucket_root / "anchor_posterior.npz"
    smoke_path = bucket_root / "anchor_posterior.smoke.npz"
    if full_path.exists():
        return full_path
    if smoke_path.exists():
        return smoke_path
    raise FileNotFoundError(
        f"Could not find `anchor_posterior.npz` or `anchor_posterior.smoke.npz` under {bucket_root}"
    )


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def multigammaln(a: np.ndarray, dim: int) -> np.ndarray:
    """Log multivariate gamma Gamma_dim(a), vectorized over `a`."""

    result = dim * (dim - 1) * 0.25 * np.log(np.pi)
    for idx in range(1, dim + 1):
        result = result + gammaln(a + (1.0 - idx) * 0.5)
    return result


def mvdigamma(a: np.ndarray, dim: int) -> np.ndarray:
    """Multivariate digamma d/da log Gamma_dim(a), vectorized over `a`."""

    result = np.zeros_like(a, dtype=np.float64)
    for idx in range(1, dim + 1):
        result = result + digamma(a + (1.0 - idx) * 0.5)
    return result


def safe_logdet(matrix: np.ndarray, eps: float) -> np.ndarray:
    """Return logdet for a stack of SPD-ish matrices with diagonal jitter."""

    matrix = np.asarray(matrix, dtype=np.float64)
    dim = matrix.shape[-1]
    eye = np.eye(dim, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(matrix + eps * eye)
    if np.any(sign <= 0):
        repaired = np.array(matrix, copy=True)
        repaired += (eps * 100.0) * eye
        sign, logdet = np.linalg.slogdet(repaired)
    return np.where(sign > 0, logdet, np.nan)


def normal_wishart_entropy(kappa: np.ndarray, u: np.ndarray, n: np.ndarray, eps: float) -> np.ndarray:
    """Entropy H[q(mu, precision)] for VBGS's Normal-Wishart posterior.

    VBGS parameterization:
      precision Lambda ~ Wishart(n, U)
      mu | Lambda ~ Normal(mean, (kappa Lambda)^-1)
      E[Lambda] = n U
    """

    kappa = np.clip(np.asarray(kappa, dtype=np.float64).reshape(kappa.shape[0], -1)[:, 0], eps, np.inf)
    n = np.clip(np.asarray(n, dtype=np.float64).reshape(n.shape[0], -1)[:, 0], eps, np.inf)
    u = np.asarray(u, dtype=np.float64)
    dim = int(u.shape[-1])

    logdet_u = safe_logdet(u, eps)
    e_logdet_precision = dim * np.log(2.0) + logdet_u + mvdigamma(n * 0.5, dim)

    wishart_entropy = (
        -0.5 * (n - dim - 1.0) * e_logdet_precision
        + 0.5 * n * dim
        + 0.5 * n * dim * np.log(2.0)
        + 0.5 * n * logdet_u
        + multigammaln(n * 0.5, dim)
    )
    normal_given_precision_entropy = 0.5 * (
        dim * (1.0 + np.log(2.0 * np.pi)) - dim * np.log(kappa) - e_logdet_precision
    )
    return wishart_entropy + normal_given_precision_entropy


def delta_mvn_entropy(kappa: np.ndarray, u: np.ndarray, n: np.ndarray, eps: float) -> np.ndarray:
    """Entropy of the delta mean using covariance (kappa * E[precision])^-1."""

    kappa = np.clip(np.asarray(kappa, dtype=np.float64).reshape(kappa.shape[0], -1)[:, 0], eps, np.inf)
    n = np.clip(np.asarray(n, dtype=np.float64).reshape(n.shape[0], -1)[:, 0], eps, np.inf)
    u = np.asarray(u, dtype=np.float64)
    dim = int(u.shape[-1])
    logdet_precision = dim * np.log(n) + safe_logdet(u, eps)
    return 0.5 * (dim * (1.0 + np.log(2.0 * np.pi)) - dim * np.log(kappa) - logdet_precision)


def dirichlet_entropy(alpha: np.ndarray, eps: float) -> np.ndarray:
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), eps, np.inf)
    alpha0 = alpha.sum(axis=-1)
    k = alpha.shape[-1]
    log_beta = np.sum(gammaln(alpha), axis=-1) - gammaln(alpha0)
    return log_beta + (alpha0 - k) * digamma(alpha0) - np.sum((alpha - 1.0) * digamma(alpha), axis=-1)


def compute_uncertainty(posterior: Dict[str, np.ndarray], *, eps: float, include_dirichlet_entropy: bool) -> Dict[str, np.ndarray]:
    is_observed = np.asarray(posterior["is_observed"], dtype=bool)
    observed_anchor_ids = np.asarray(posterior["observed_anchor_ids"], dtype=np.int64)
    final_k = np.asarray(posterior["final_k"], dtype=np.int32)
    fit_completed = np.asarray(
        posterior.get("fit_completed", np.isfinite(posterior["final_elbo"])),
        dtype=bool,
    )
    n_anchors = int(is_observed.shape[0])
    n_rows = int(observed_anchor_ids.shape[0])

    u_observed = np.full((n_rows,), np.nan, dtype=np.float64)
    spatial_entropy_observed = np.full((n_rows,), np.nan, dtype=np.float64)
    delta_entropy_observed = np.full((n_rows,), np.nan, dtype=np.float64)
    dirichlet_entropy_observed = np.full((n_rows,), np.nan, dtype=np.float64)
    effective_components = np.zeros((n_rows,), dtype=np.int32)

    for row_idx in range(n_rows):
        if not fit_completed[row_idx] or final_k[row_idx] <= 0:
            continue
        k = int(final_k[row_idx])
        effective_components[row_idx] = k

        alpha = np.asarray(posterior["alpha"][row_idx, :k], dtype=np.float64)
        weights = alpha / np.clip(alpha.sum(), eps, np.inf)

        spatial_entropy = normal_wishart_entropy(
            posterior["spatial_kappa"][row_idx, :k],
            posterior["spatial_u"][row_idx, :k],
            posterior["spatial_n"][row_idx, :k],
            eps,
        )
        delta_entropy = delta_mvn_entropy(
            posterior["delta_kappa"][row_idx, :k],
            posterior["delta_u"][row_idx, :k],
            posterior["delta_n"][row_idx, :k],
            eps,
        )
        dir_entropy = float(dirichlet_entropy(alpha[None, :], eps)[0])
        combined_components = spatial_entropy + delta_entropy
        u_value = float(np.sum(weights * combined_components))
        if include_dirichlet_entropy:
            u_value += dir_entropy

        u_observed[row_idx] = u_value
        spatial_entropy_observed[row_idx] = float(np.sum(weights * spatial_entropy))
        delta_entropy_observed[row_idx] = float(np.sum(weights * delta_entropy))
        dirichlet_entropy_observed[row_idx] = dir_entropy

    u = np.full((n_anchors,), np.nan, dtype=np.float64)
    completed_anchor_ids = observed_anchor_ids[fit_completed]
    u[completed_anchor_ids] = u_observed[fit_completed]

    return {
        "U": u,
        "u_observed": u_observed,
        "spatial_entropy_observed": spatial_entropy_observed,
        "delta_entropy_observed": delta_entropy_observed,
        "dirichlet_entropy_observed": dirichlet_entropy_observed,
        "effective_components": effective_components,
        "is_observed": is_observed,
        "fit_completed": fit_completed,
        "observed_anchor_ids": observed_anchor_ids,
    }


def fill_unobserved(u: np.ndarray, fitted_mask: np.ndarray, explicit_u_max: float | None) -> Tuple[np.ndarray, float]:
    finite_fitted = np.isfinite(u) & fitted_mask
    if explicit_u_max is not None:
        u_max = float(explicit_u_max)
    elif finite_fitted.any():
        u_max = float(np.nanmax(u[finite_fitted]))
    else:
        u_max = 1.0
    filled = np.array(u, copy=True)
    filled[~finite_fitted] = u_max
    return filled.astype(np.float32), u_max


def main() -> None:
    args = parse_args()
    bucket_root = resolve_bucket_root(args)
    posterior_path = resolve_posterior_path(args, bucket_root)
    output_root = (args.output_root or posterior_path.parent).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    posterior = load_npz(posterior_path)
    components = compute_uncertainty(
        posterior,
        eps=args.eps,
        include_dirichlet_entropy=args.include_dirichlet_entropy,
    )

    n_anchors = int(components["U"].shape[0])
    fitted_anchor_mask = np.zeros((n_anchors,), dtype=bool)
    fitted_anchor_mask[components["observed_anchor_ids"][components["fit_completed"]]] = True
    u_filled, u_max = fill_unobserved(components["U"], fitted_anchor_mask, args.u_max)

    output_path = output_root / args.output_name
    components_path = output_root / args.components_name
    metadata_path = output_root / args.metadata_name

    np.save(output_path, u_filled)
    finite_fitted = u_filled[fitted_anchor_mask]
    if finite_fitted.size and args.histogram_bins > 0:
        hist_counts, hist_edges = np.histogram(finite_fitted, bins=args.histogram_bins)
    else:
        hist_counts = np.zeros((0,), dtype=np.int64)
        hist_edges = np.zeros((0,), dtype=np.float32)

    np.savez_compressed(
        components_path,
        U_raw=components["U"].astype(np.float32),
        U=u_filled,
        u_observed=components["u_observed"].astype(np.float32),
        spatial_entropy_observed=components["spatial_entropy_observed"].astype(np.float32),
        delta_entropy_observed=components["delta_entropy_observed"].astype(np.float32),
        dirichlet_entropy_observed=components["dirichlet_entropy_observed"].astype(np.float32),
        effective_components=components["effective_components"],
        observed_anchor_ids=components["observed_anchor_ids"],
        fit_completed=components["fit_completed"],
        fitted_anchor_mask=fitted_anchor_mask,
        u_max=np.array(u_max, dtype=np.float32),
        histogram_counts=hist_counts,
        histogram_edges=hist_edges.astype(np.float32),
    )

    metadata = {
        "drive": args.drive,
        "posterior_path": str(posterior_path),
        "output_U": str(output_path),
        "components_path": str(components_path),
        "anchor_count": n_anchors,
        "observed_anchor_rows": int(components["observed_anchor_ids"].shape[0]),
        "fitted_anchor_count": int(fitted_anchor_mask.sum()),
        "u_max": u_max,
        "include_dirichlet_entropy": bool(args.include_dirichlet_entropy),
        "u_fitted_min": float(np.min(finite_fitted)) if finite_fitted.size else None,
        "u_fitted_p50": float(np.percentile(finite_fitted, 50)) if finite_fitted.size else None,
        "u_fitted_p90": float(np.percentile(finite_fitted, 90)) if finite_fitted.size else None,
        "u_fitted_p99": float(np.percentile(finite_fitted, 99)) if finite_fitted.size else None,
        "u_fitted_max": float(np.max(finite_fitted)) if finite_fitted.size else None,
        "histogram_bins": int(args.histogram_bins),
    }
    save_json(metadata_path, metadata)

    print(f"Wrote {output_path}")
    print(f"Wrote {components_path}")
    print(f"Wrote {metadata_path}")
    print(
        f"Fitted anchors: {metadata['fitted_anchor_count']:,} / {n_anchors:,} | "
        f"U range={metadata['u_fitted_min']} .. {metadata['u_fitted_max']} | U_MAX={u_max}"
    )


if __name__ == "__main__":
    main()
