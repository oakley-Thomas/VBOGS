#!/usr/bin/env python3

"""Compute per-anchor scalar uncertainty from fitted VBGS posteriors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
        help="Drive id used to resolve default paths.",
    )
    parser.add_argument(
        "--bucket-root",
        type=Path,
        default=None,
        help="Directory containing M4 artifacts. Defaults to `data/m4/<drive>`.",
    )
    parser.add_argument(
        "--posterior",
        type=Path,
        default=None,
        help=(
            "Posterior artifact to read. Defaults to full fit "
            "`anchor_posterior.npz`, then smoke fit."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output uncertainty array. Defaults to `<bucket-root>/U.npy`.",
    )
    parser.add_argument(
        "--components-output",
        type=Path,
        default=None,
        help=(
            "Optional diagnostic component artifact. Defaults to "
            "`<bucket-root>/uncertainty_components.npz`."
        ),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Metadata JSON path. Defaults to `<bucket-root>/uncertainty_metadata.json`.",
    )
    parser.add_argument(
        "--histogram-output",
        type=Path,
        default=None,
        help="Histogram PNG path. Defaults to `<bucket-root>/uncertainty_histogram.png`.",
    )
    parser.add_argument(
        "--no-histogram",
        action="store_true",
        help="Skip writing the uncertainty histogram PNG.",
    )
    parser.add_argument(
        "--u-max",
        type=float,
        default=None,
        help=(
            "Value assigned to unobserved anchors. Defaults to the maximum "
            "finite observed uncertainty."
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0e-8,
        help="Small positive floor used for logs and matrix eigenvalues.",
    )
    return parser.parse_args()


def resolve_bucket_root(drive: str, bucket_root: Path | None) -> Path:
    if bucket_root is not None:
        return bucket_root.resolve()
    return (REPO_ROOT / "data" / "m4" / drive).resolve()


def resolve_posterior_path(bucket_root: Path, posterior: Path | None) -> Path:
    if posterior is not None:
        return posterior.resolve()
    full = bucket_root / "anchor_posterior.npz"
    if full.exists():
        return full
    smoke = bucket_root / "anchor_posterior.smoke.npz"
    if smoke.exists():
        return smoke
    raise FileNotFoundError(
        f"Could not find `anchor_posterior.npz` or `anchor_posterior.smoke.npz` under {bucket_root}"
    )


def mvgammaln(x: np.ndarray, dim: int) -> np.ndarray:
    offsets = 0.5 * np.arange(dim, dtype=np.float64)
    return (
        np.sum(gammaln(np.expand_dims(x, axis=-1) - offsets), axis=-1)
        + dim * (dim - 1) * 0.25 * np.log(np.pi)
    )


def mvdigamma(x: np.ndarray, dim: int) -> np.ndarray:
    offsets = 0.5 * np.arange(dim, dtype=np.float64)
    return np.sum(digamma(np.expand_dims(x, axis=-1) - offsets), axis=-1)


def logdet_posdef(mats: np.ndarray, eps: float) -> tuple[np.ndarray, int]:
    mats = 0.5 * (mats + np.swapaxes(mats, -1, -2))
    sign, logdet = np.linalg.slogdet(mats)
    bad = (~np.isfinite(logdet)) | (sign <= 0)
    fallback_count = int(np.count_nonzero(bad))
    if fallback_count:
        eigvals = np.linalg.eigvalsh(mats[bad])
        logdet = logdet.copy()
        logdet[bad] = np.sum(np.log(np.clip(eigvals, eps, None)), axis=-1)
    return logdet, fallback_count


def normal_wishart_entropy(
    kappa: np.ndarray,
    u: np.ndarray,
    n: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, int]:
    """Joint entropy of Normal-Wishart q(mu, Lambda)."""

    dim = int(u.shape[-1])
    kappa = np.clip(np.asarray(kappa, dtype=np.float64), eps, None)
    n = np.clip(np.asarray(n, dtype=np.float64), dim + eps, None)
    logdet_u, fallback_count = logdet_posdef(np.asarray(u, dtype=np.float64), eps)

    expected_logdet_lambda = dim * np.log(2.0) + logdet_u + mvdigamma(n / 2.0, dim)
    wishart_entropy = (
        -0.5 * (n - dim - 1.0) * expected_logdet_lambda
        + 0.5 * n * dim
        + 0.5 * n * dim * np.log(2.0)
        + 0.5 * n * logdet_u
        + mvgammaln(n / 2.0, dim)
    )
    normal_entropy = (
        0.5 * dim * (1.0 + np.log(2.0 * np.pi))
        - 0.5 * dim * np.log(kappa)
        - 0.5 * expected_logdet_lambda
    )
    return wishart_entropy + normal_entropy, fallback_count


def delta_mvn_entropy(
    kappa: np.ndarray,
    u: np.ndarray,
    n: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, int]:
    """Entropy of the delta posterior mean using E[precision] = n * U."""

    dim = int(u.shape[-1])
    kappa = np.clip(np.asarray(kappa, dtype=np.float64), eps, None)
    n = np.clip(np.asarray(n, dtype=np.float64), eps, None)
    logdet_u, fallback_count = logdet_posdef(np.asarray(u, dtype=np.float64), eps)
    logdet_cov = -logdet_u - dim * np.log(n) - dim * np.log(kappa)
    entropy = 0.5 * (dim * (1.0 + np.log(2.0 * np.pi)) + logdet_cov)
    return entropy, fallback_count


def dirichlet_entropy(alpha: np.ndarray, *, eps: float) -> np.ndarray:
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), eps, None)
    alpha0 = np.sum(alpha, axis=-1)
    k = alpha.shape[-1]
    log_b = np.sum(gammaln(alpha), axis=-1) - gammaln(alpha0)
    return log_b + (alpha0 - k) * digamma(alpha0) - np.sum(
        (alpha - 1.0) * digamma(alpha),
        axis=-1,
    )


def compute_uncertainty(posterior: Any, *, u_max: float | None, eps: float) -> dict[str, np.ndarray | float | int]:
    is_observed = np.asarray(posterior["is_observed"], dtype=bool)
    observed_anchor_ids = np.asarray(posterior["observed_anchor_ids"], dtype=np.int64)
    final_k = np.asarray(posterior["final_k"], dtype=np.int32)
    fit_completed = np.asarray(
        posterior["fit_completed"] if "fit_completed" in posterior else final_k > 0,
        dtype=bool,
    )

    anchor_count = int(is_observed.shape[0])
    observed_count = int(observed_anchor_ids.shape[0])
    if final_k.shape[0] != observed_count:
        raise ValueError(
            f"`final_k` length {final_k.shape[0]} does not match observed rows {observed_count}"
        )

    alpha = np.asarray(posterior["alpha"], dtype=np.float64)
    spatial_kappa = np.squeeze(np.asarray(posterior["spatial_kappa"], dtype=np.float64), axis=(-1, -2))
    spatial_u = np.asarray(posterior["spatial_u"], dtype=np.float64)
    spatial_n = np.squeeze(np.asarray(posterior["spatial_n"], dtype=np.float64), axis=(-1, -2))
    delta_kappa = np.squeeze(np.asarray(posterior["delta_kappa"], dtype=np.float64), axis=(-1, -2))
    delta_u = np.asarray(posterior["delta_u"], dtype=np.float64)
    delta_n = np.squeeze(np.asarray(posterior["delta_n"], dtype=np.float64), axis=(-1, -2))

    if alpha.shape[0] != observed_count:
        raise ValueError(
            f"`alpha` row count {alpha.shape[0]} does not match observed rows {observed_count}"
        )

    max_k = int(alpha.shape[1])
    if fit_completed.shape[0] != observed_count:
        raise ValueError(
            f"`fit_completed` length {fit_completed.shape[0]} does not match observed rows {observed_count}"
        )

    active_mask = (np.arange(max_k)[None, :] < final_k[:, None]) & fit_completed[:, None]
    matrix_active_mask = active_mask[:, :, None, None]
    safe_spatial_u = np.where(
        matrix_active_mask,
        spatial_u,
        np.eye(spatial_u.shape[-1], dtype=np.float64),
    )
    safe_delta_u = np.where(
        matrix_active_mask,
        delta_u,
        np.eye(delta_u.shape[-1], dtype=np.float64),
    )
    safe_spatial_kappa = np.where(active_mask, spatial_kappa, 1.0)
    safe_spatial_n = np.where(active_mask, spatial_n, spatial_u.shape[-1] + 2.0)
    safe_delta_kappa = np.where(active_mask, delta_kappa, 1.0)
    safe_delta_n = np.where(active_mask, delta_n, delta_u.shape[-1] + 2.0)

    spatial_entropy, spatial_fallbacks = normal_wishart_entropy(
        safe_spatial_kappa,
        safe_spatial_u,
        safe_spatial_n,
        eps=eps,
    )
    delta_entropy, delta_fallbacks = delta_mvn_entropy(
        safe_delta_kappa,
        safe_delta_u,
        safe_delta_n,
        eps=eps,
    )
    component_entropy = spatial_entropy + delta_entropy

    active_alpha = np.where(active_mask, alpha, 0.0)
    alpha_sum = np.sum(active_alpha, axis=1, keepdims=True)
    weights = np.divide(
        active_alpha,
        np.clip(alpha_sum, eps, None),
        out=np.zeros_like(active_alpha),
        where=alpha_sum > 0,
    )

    observed_u = np.full(observed_count, np.nan, dtype=np.float64)
    observed_u[fit_completed] = np.sum(
        np.where(active_mask, weights * component_entropy, 0.0),
        axis=1,
    )[fit_completed]
    finite_observed = observed_u[np.isfinite(observed_u)]
    if finite_observed.size == 0:
        default_u_max = 1.0
    else:
        default_u_max = float(np.max(finite_observed))
    unobserved_value = float(default_u_max if u_max is None else u_max)

    uncertainty = np.full(anchor_count, unobserved_value, dtype=np.float32)
    uncertainty[observed_anchor_ids[fit_completed]] = observed_u[fit_completed].astype(np.float32)

    dirichlet_h = np.full(observed_count, np.nan, dtype=np.float32)
    for row in range(observed_count):
        k = int(final_k[row])
        if k > 0:
            dirichlet_h[row] = np.float32(dirichlet_entropy(alpha[row, :k], eps=eps))

    return {
        "uncertainty": uncertainty,
        "observed_uncertainty": observed_u.astype(np.float32),
        "fit_completed": fit_completed,
        "weights": weights.astype(np.float32),
        "spatial_entropy": np.where(active_mask, spatial_entropy, np.nan).astype(np.float32),
        "delta_entropy": np.where(active_mask, delta_entropy, np.nan).astype(np.float32),
        "component_entropy": np.where(active_mask, component_entropy, np.nan).astype(np.float32),
        "dirichlet_entropy": dirichlet_h,
        "unobserved_value": unobserved_value,
        "spatial_logdet_fallbacks": spatial_fallbacks,
        "delta_logdet_fallbacks": delta_fallbacks,
    }


def write_histogram(path: Path, uncertainty: np.ndarray, is_observed: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    finite = uncertainty[np.isfinite(uncertainty)]
    observed = uncertainty[is_observed & np.isfinite(uncertainty)]
    plt.figure(figsize=(8, 5))
    if finite.size:
        plt.hist(finite, bins=80, alpha=0.55, label="all anchors")
    if observed.size:
        plt.hist(observed, bins=80, alpha=0.55, label="observed anchors")
    plt.xlabel("uncertainty")
    plt.ylabel("anchor count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


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


def main() -> None:
    args = parse_args()
    bucket_root = resolve_bucket_root(args.drive, args.bucket_root)
    posterior_path = resolve_posterior_path(bucket_root, args.posterior)
    output_path = (args.output or (bucket_root / "U.npy")).resolve()
    components_path = (args.components_output or (bucket_root / "uncertainty_components.npz")).resolve()
    metadata_path = (args.metadata_output or (bucket_root / "uncertainty_metadata.json")).resolve()
    histogram_path = (args.histogram_output or (bucket_root / "uncertainty_histogram.png")).resolve()

    print(f"Loading posterior: {posterior_path}")
    posterior = np.load(posterior_path)
    result = compute_uncertainty(posterior, u_max=args.u_max, eps=args.eps)

    uncertainty = np.asarray(result["uncertainty"], dtype=np.float32)
    is_observed = np.asarray(posterior["is_observed"], dtype=bool)
    observed_anchor_ids = np.asarray(posterior["observed_anchor_ids"], dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, uncertainty)

    components_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        components_path,
        observed_anchor_ids=observed_anchor_ids,
        observed_uncertainty=result["observed_uncertainty"],
        weights=result["weights"],
        spatial_entropy=result["spatial_entropy"],
        delta_entropy=result["delta_entropy"],
        component_entropy=result["component_entropy"],
        dirichlet_entropy=result["dirichlet_entropy"],
        unobserved_value=np.array(result["unobserved_value"], dtype=np.float32),
    )

    if not args.no_histogram:
        write_histogram(histogram_path, uncertainty, is_observed)

    metadata = {
        "drive": args.drive,
        "posterior_path": str(posterior_path),
        "output_path": str(output_path),
        "components_path": str(components_path),
        "histogram_path": None if args.no_histogram else str(histogram_path),
        "anchor_count": int(uncertainty.shape[0]),
        "observed_anchor_count": int(observed_anchor_ids.shape[0]),
        "unobserved_anchor_count": int(uncertainty.shape[0] - observed_anchor_ids.shape[0]),
        "unobserved_value": float(result["unobserved_value"]),
        "spatial_logdet_fallbacks": int(result["spatial_logdet_fallbacks"]),
        "delta_logdet_fallbacks": int(result["delta_logdet_fallbacks"]),
        "all_summary": summarize(uncertainty),
        "observed_summary": summarize(uncertainty[observed_anchor_ids]),
    }
    save_json(metadata_path, metadata)

    print(f"Wrote {output_path}")
    print(f"Wrote {components_path}")
    print(f"Wrote {metadata_path}")
    if not args.no_histogram:
        print(f"Wrote {histogram_path}")
    print(
        "Uncertainty summary: "
        + json.dumps(metadata["all_summary"], sort_keys=True)
    )


if __name__ == "__main__":
    main()
