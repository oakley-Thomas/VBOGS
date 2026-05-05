import numpy as np

from scripts.compute_uncertainty import compute_uncertainty, dirichlet_entropy


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
