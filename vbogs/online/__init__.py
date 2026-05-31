"""Online VBOGS helpers.

The online package keeps reusable real-time pieces independent from ROS2,
Torch, and JAX process entry points. Scripts can import these functions in
plain unit tests without requiring a live ROS installation.
"""

from .bucketing import (
    AnchorGridCache,
    bucket_points_with_cache,
    build_anchor_grid_cache,
    load_anchor_grid_cache,
    save_anchor_grid_cache,
)
from .normalization import normalize_online_observations
from .scoring import CandidateScore, rank_candidate_scores, score_uncertainty_alpha
from .state import (
    ONLINE_STATE_VERSION,
    apply_online_exact_fixed_k_update,
    apply_online_moment_update,
    atomic_save_npy,
    atomic_save_npz,
    backfill_initial_fields_from_points,
    expand_posterior_to_anchor_rows,
    load_npz_dict,
)

__all__ = [
    "AnchorGridCache",
    "CandidateScore",
    "ONLINE_STATE_VERSION",
    "apply_online_exact_fixed_k_update",
    "apply_online_moment_update",
    "atomic_save_npy",
    "atomic_save_npz",
    "backfill_initial_fields_from_points",
    "bucket_points_with_cache",
    "build_anchor_grid_cache",
    "expand_posterior_to_anchor_rows",
    "load_anchor_grid_cache",
    "load_npz_dict",
    "normalize_online_observations",
    "rank_candidate_scores",
    "save_anchor_grid_cache",
    "score_uncertainty_alpha",
]
