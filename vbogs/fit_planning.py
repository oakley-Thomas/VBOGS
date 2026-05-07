"""Planning helpers for exact batched VBGS anchor fitting."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BucketPlan:
    effective_buckets: tuple[int, ...]
    count_to_bucket: np.ndarray
    bucket_sizes: tuple[int, ...]
    anchors_per_bucket: tuple[int, ...]
    group_caps: tuple[int, ...]
    estimated_fit_calls: int
    exact_overflow_bucket_count: int
    exact_overflow_anchor_count: int


def parse_batch_buckets(raw: str, batch_size: int) -> tuple[int, ...]:
    buckets = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not buckets:
        raise ValueError("--batch-buckets must contain at least one positive integer")
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError("--batch-buckets values must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if buckets[-1] < batch_size:
        buckets.append(batch_size)
    return tuple(buckets)


def bucket_for_count(count: int, buckets: Sequence[int]) -> int:
    for bucket in buckets:
        if count <= bucket:
            return int(bucket)
    return int(count)


def extend_batch_buckets(
    buckets: Sequence[int],
    max_count: int,
    *,
    enabled: bool = True,
    growth_factor: float = 1.5,
    round_to: int = 256,
) -> tuple[int, ...]:
    """Append rounded geometric tail buckets up to `max_count`."""

    if not buckets:
        raise ValueError("buckets must contain at least one value")
    if any(int(bucket) <= 0 for bucket in buckets):
        raise ValueError("bucket values must be positive")
    if growth_factor <= 1.0:
        raise ValueError("growth_factor must be greater than 1")
    if round_to <= 0:
        raise ValueError("round_to must be positive")

    result = sorted({int(bucket) for bucket in buckets})
    if not enabled or max_count <= result[-1]:
        return tuple(result)

    while result[-1] < max_count:
        next_bucket = int(ceil((result[-1] * growth_factor) / round_to) * round_to)
        next_bucket = max(next_bucket, result[-1] + round_to)
        result.append(next_bucket)
    return tuple(result)


def compute_bucket_group_size(
    *,
    bucket_size: int,
    vmap_group_size: int,
    max_padded_points_per_group: int,
) -> int:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    if vmap_group_size <= 0:
        raise ValueError("--vmap-group-size must be positive")
    if max_padded_points_per_group <= 0:
        raise ValueError("--max-padded-points-per-group must be positive")
    return min(vmap_group_size, max(1, max_padded_points_per_group // bucket_size))


def build_bucket_plan(
    point_counts: np.ndarray,
    configured_buckets: Sequence[int],
    *,
    auto_extend_buckets: bool,
    vmap_group_size: int,
    max_padded_points_per_group: int,
) -> BucketPlan:
    observed_counts = np.asarray(point_counts, dtype=np.int64).reshape(-1)
    max_count = int(observed_counts.max()) if observed_counts.size else 0
    effective_buckets = extend_batch_buckets(
        configured_buckets,
        max_count,
        enabled=auto_extend_buckets,
    )
    count_to_bucket = np.array(
        [bucket_for_count(int(count), effective_buckets) for count in observed_counts],
        dtype=np.int32,
    )

    if count_to_bucket.size:
        bucket_sizes_arr, anchors_arr = np.unique(count_to_bucket, return_counts=True)
    else:
        bucket_sizes_arr = np.zeros((0,), dtype=np.int32)
        anchors_arr = np.zeros((0,), dtype=np.int64)

    bucket_sizes = tuple(int(value) for value in bucket_sizes_arr.tolist())
    anchors_per_bucket = tuple(int(value) for value in anchors_arr.tolist())
    group_caps = tuple(
        compute_bucket_group_size(
            bucket_size=bucket_size,
            vmap_group_size=vmap_group_size,
            max_padded_points_per_group=max_padded_points_per_group,
        )
        for bucket_size in bucket_sizes
    )
    estimated_fit_calls = sum(
        int(ceil(anchor_count / group_cap))
        for anchor_count, group_cap in zip(anchors_per_bucket, group_caps)
    )
    effective_set = set(effective_buckets)
    overflow_mask = np.array([bucket not in effective_set for bucket in count_to_bucket], dtype=bool)
    exact_overflow_bucket_count = len({int(bucket) for bucket in count_to_bucket[overflow_mask].tolist()})
    exact_overflow_anchor_count = int(overflow_mask.sum())

    return BucketPlan(
        effective_buckets=effective_buckets,
        count_to_bucket=count_to_bucket,
        bucket_sizes=bucket_sizes,
        anchors_per_bucket=anchors_per_bucket,
        group_caps=group_caps,
        estimated_fit_calls=estimated_fit_calls,
        exact_overflow_bucket_count=exact_overflow_bucket_count,
        exact_overflow_anchor_count=exact_overflow_anchor_count,
    )
