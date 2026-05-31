"""Candidate scoring helpers for rendered VBOGS uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CandidateScore:
    candidate_index: int
    score: float
    unc_sum: float
    alpha_sum: float
    stale_update: bool = False
    metadata: dict | None = None


def score_uncertainty_alpha(
    unc_image: np.ndarray,
    alpha_image: np.ndarray,
    *,
    eps: float = 1.0e-6,
) -> tuple[float, float, float]:
    """Return alpha-normalized uncertainty score and image sums."""

    unc_sum = float(np.asarray(unc_image, dtype=np.float64).sum())
    alpha_sum = float(np.asarray(alpha_image, dtype=np.float64).sum())
    score = unc_sum / (alpha_sum + float(eps))
    return score, unc_sum, alpha_sum


def rank_candidate_scores(scores: Iterable[CandidateScore]) -> list[CandidateScore]:
    return sorted(scores, key=lambda item: item.score, reverse=True)
