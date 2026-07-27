import numpy as np
import pytest

from scripts.compute_uncertainty import (
    compute_uncertainty,
    delta_kl,
    dirichlet_entropy,
    dirichlet_kl,
    normal_wishart_kl,
)


def posterior_with_priors(*, completed=True, sharpened=False):
    kappa = 4.0 if sharpened else 1.0e-3
    u_scale = 1.0 if sharpened else 22500.0
    n = 8.0 if sharpened else 5.0
    return {
        "is_observed": np.array([False, True, True, False]),
        "observed_anchor_ids": np.array([1, 2], dtype=np.int64),
        "final_k": np.array([1, 1], dtype=np.int32),
        "fit_completed": np.array([True, completed]),
        "k_max": np.array(1, dtype=np.int16),
        "alpha": np.ones((2, 1), dtype=np.float32),
        "spatial_mean": np.zeros((2, 1, 3, 1), dtype=np.float32),
        "spatial_kappa": np.full((2, 1, 1, 1), kappa, dtype=np.float32),
        "spatial_u": np.broadcast_to(np.eye(3, dtype=np.float32) * u_scale, (2, 1, 3, 3)).copy(),
        "spatial_n": np.full((2, 1, 1, 1), n, dtype=np.float32),
        "delta_mean": np.zeros((2, 1, 3, 1), dtype=np.float32),
        "delta_kappa": np.full((2, 1, 1, 1), 1.0e-2 if not sharpened else 5.0, dtype=np.float32),
        "delta_u": np.broadcast_to(np.eye(3, dtype=np.float32) * 1.0e8, (2, 1, 3, 3)).copy(),
        "delta_n": np.full((2, 1, 1, 1), 5.0, dtype=np.float32),
        "prior_alpha": np.ones(1, dtype=np.float32),
        "prior_spatial_mean": np.zeros((3, 1), dtype=np.float32),
        "prior_spatial_kappa": np.array([[1.0e-3]], dtype=np.float32),
        "prior_spatial_u": np.eye(3, dtype=np.float32) * 22500.0,
        "prior_spatial_n": np.array([[5.0]], dtype=np.float32),
        "prior_delta_mean": np.zeros((3, 1), dtype=np.float32),
        "prior_delta_kappa": np.array([[1.0e-2]], dtype=np.float32),
        "prior_delta_u": np.eye(3, dtype=np.float32) * 1.0e8,
        "prior_delta_n": np.array([[5.0]], dtype=np.float32),
    }


def test_compute_uncertainty_writes_full_anchor_vector_with_unobserved_max():
    posterior = {
        "is_observed": np.array([False, True, False]),
        "observed_anchor_ids": np.array([1], dtype=np.int64),
        "final_k": np.array([1], dtype=np.int32),
        "alpha": np.array([[2.0, np.nan]], dtype=np.float32),
        "spatial_kappa": np.array([[[[4.0]], [[np.nan]]]], dtype=np.float32),
        "spatial_u": np.array(
            [[np.eye(3, dtype=np.float32), np.full((3, 3), np.nan, dtype=np.float32)]]
        ),
        "spatial_n": np.array([[[[8.0]], [[np.nan]]]], dtype=np.float32),
        "delta_kappa": np.array([[[[5.0]], [[np.nan]]]], dtype=np.float32),
        "delta_u": np.array(
            [[np.eye(3, dtype=np.float32), np.full((3, 3), np.nan, dtype=np.float32)]]
        ),
        "delta_n": np.array([[[[6.0]], [[np.nan]]]], dtype=np.float32),
    }

    result = compute_uncertainty(posterior, u_max=None, eps=1.0e-8)
    uncertainty = result["uncertainty"]

    assert uncertainty.shape == (3,)
    assert np.isfinite(uncertainty).all()
    assert uncertainty[0] == uncertainty[1]
    assert uncertainty[2] == uncertainty[1]
    assert result["weights"].shape == (1, 2)
    assert result["weights"][0, 0] == 1.0


def test_dirichlet_entropy_is_finite_for_positive_alpha():
    entropy = dirichlet_entropy(np.array([1.0, 2.0, 3.0]), eps=1.0e-8)

    assert np.isfinite(entropy)


def test_kl_uncertainty_uses_prior_as_one_and_marks_only_completed_fits_observed():
    result = compute_uncertainty(posterior_with_priors(completed=False), u_max=None, eps=1.0e-8)

    assert result["uncertainty_kl"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert result["observed_mask"].tolist() == [False, True, False, False]
    assert result["information_gain"][0] == 0.0
    assert result["information_gain"][2] == 0.0


def test_sharper_posterior_has_more_information_and_lower_kl_uncertainty():
    prior_result = compute_uncertainty(posterior_with_priors(), u_max=None, eps=1.0e-8)
    sharp_result = compute_uncertainty(
        posterior_with_priors(sharpened=True), u_max=None, eps=1.0e-8
    )

    assert sharp_result["information_gain"][1] > prior_result["information_gain"][1]
    assert sharp_result["uncertainty_kl"][1] < prior_result["uncertainty_kl"][1]


def test_numpy_kl_helpers_match_vbgs_conjugate_models():
    pytest.importorskip("jax")
    from pathlib import Path

    import jax.numpy as jnp
    import jax.random as jr
    from scripts.fit_anchors import add_vbgs_to_path, make_batched_volume_delta_mixture

    add_vbgs_to_path(Path(__file__).resolve().parents[1] / "vbgs")
    from vbgs.model.model import DeltaMixture
    from vbgs.vi.conjugate.multinomial import Multinomial
    from vbgs.vi.conjugate.mvn import MultivariateNormal
    from vbgs.vi.models.mixture import Mixture
    from vbgs.vi.utils import ArrayDict

    model = make_batched_volume_delta_mixture(
        key=jr.PRNGKey(0), n_components=2,
        mean_init=jnp.ones((1, 2, 6, 1), dtype=jnp.float32), n_anchors=1,
        MultivariateNormal=MultivariateNormal, Multinomial=Multinomial,
        Mixture=Mixture, DeltaMixture=DeltaMixture, ArrayDict=ArrayDict,
    )
    spatial, _ = normal_wishart_kl(
        model.mixture.likelihood.kappa, model.mixture.likelihood.mean,
        model.mixture.likelihood.u, model.mixture.likelihood.n,
        model.mixture.likelihood.prior_kappa, model.mixture.likelihood.prior_mean,
        jnp.linalg.inv(model.mixture.likelihood.prior_inv_u), model.mixture.likelihood.prior_n, eps=1.0e-8,
    )
    delta, _ = delta_kl(
        model.delta.kappa, model.delta.mean, model.delta.u, model.delta.n,
        model.delta.prior_kappa, model.delta.prior_mean,
        jnp.linalg.inv(model.delta.prior_inv_u), model.delta.prior_n, eps=1.0e-8,
    )
    mixture = dirichlet_kl(model.mixture.prior.alpha, model.mixture.prior.prior_alpha, eps=1.0e-8)
    np.testing.assert_allclose(spatial, np.asarray(model.mixture.likelihood.kl_divergence()), rtol=1.0e-4)
    np.testing.assert_allclose(delta, np.asarray(model.delta.kl_divergence()), rtol=1.0e-4)
    np.testing.assert_allclose(mixture, np.asarray(model.mixture.prior.kl_divergence()), rtol=1.0e-4)
