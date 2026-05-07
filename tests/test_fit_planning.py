import numpy as np

from scripts.run_drive_pipeline import JAX_SERVICE, build_parser, build_steps
from vbogs.fit_planning import (
    build_bucket_plan,
    compute_bucket_group_size,
    extend_batch_buckets,
    parse_batch_buckets,
)


def test_parse_batch_buckets_appends_batch_size_when_needed():
    assert parse_batch_buckets("64,128,64", 5000) == (64, 128, 5000)


def test_extend_batch_buckets_groups_dense_tail():
    buckets = extend_batch_buckets((64, 128, 256, 512, 1024, 2048, 4096, 5000), 6246)

    assert buckets[-1] >= 6246
    assert 5520 <= buckets[-1]
    assert 6246 <= buckets[-1]


def test_compute_bucket_group_size_uses_padded_point_budget():
    assert (
        compute_bucket_group_size(
            bucket_size=5000,
            vmap_group_size=64,
            max_padded_points_per_group=320000,
        )
        == 64
    )
    assert (
        compute_bucket_group_size(
            bucket_size=8000,
            vmap_group_size=64,
            max_padded_points_per_group=320000,
        )
        == 40
    )


def test_bucket_plan_avoids_exact_overflow_for_dense_tail():
    plan = build_bucket_plan(
        np.array([64, 5520, 6246], dtype=np.int32),
        parse_batch_buckets("64,128,256,512,1024,2048,4096,5000", 5000),
        auto_extend_buckets=True,
        vmap_group_size=64,
        max_padded_points_per_group=320000,
    )

    assert plan.count_to_bucket[1] == plan.count_to_bucket[2]
    assert plan.exact_overflow_bucket_count == 0
    assert plan.estimated_fit_calls == 2


def test_bucket_plan_reports_exact_overflow_when_extension_disabled():
    plan = build_bucket_plan(
        np.array([5520, 6246], dtype=np.int32),
        parse_batch_buckets("64,128,256,512,1024,2048,4096,5000", 5000),
        auto_extend_buckets=False,
        vmap_group_size=64,
        max_padded_points_per_group=320000,
    )

    assert tuple(plan.count_to_bucket.tolist()) == (5520, 6246)
    assert plan.exact_overflow_bucket_count == 2
    assert plan.exact_overflow_anchor_count == 2


def test_pipeline_forwards_fit_bucket_controls():
    parser = build_parser({})
    args = parser.parse_args(
        [
            "--drive",
            "2013_05_28_drive_0000_sync",
            "--batch-buckets",
            "64,128,5000",
            "--no-auto-extend-buckets",
            "--max-padded-points-per-group",
            "640000",
        ]
    )
    fit_step = next(step for step in build_steps(args) if step.name == "fit")

    assert fit_step.service == JAX_SERVICE
    assert "--batch-buckets" in fit_step.command
    assert "64,128,5000" in fit_step.command
    assert "--no-auto-extend-buckets" in fit_step.command
    assert "--max-padded-points-per-group" in fit_step.command
    assert "640000" in fit_step.command
