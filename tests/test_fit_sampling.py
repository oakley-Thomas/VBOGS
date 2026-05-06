import numpy as np

from vbogs.io import cap_group_counts, select_group_values


def test_select_group_values_uses_all_values_below_cap():
    offsets = np.array([0, 3], dtype=np.int64)
    values = np.array([7, 3, 5], dtype=np.int64)
    points_norm = np.arange(10 * 6, dtype=np.float32).reshape(10, 6)

    selected = select_group_values(
        offsets,
        values,
        0,
        max_values_per_group=10,
        seed=123,
    )

    assert np.array_equal(selected, values)
    assert np.array_equal(points_norm[selected], points_norm[values])


def test_select_group_values_caps_dense_group_deterministically():
    offsets = np.array([0, 20], dtype=np.int64)
    values = np.arange(20, dtype=np.int64)

    first = select_group_values(
        offsets,
        values,
        0,
        max_values_per_group=8,
        seed=123,
    )
    second = select_group_values(
        offsets,
        values,
        0,
        max_values_per_group=8,
        seed=123,
    )

    assert first.shape == (8,)
    assert np.array_equal(first, second)
    assert np.all(np.diff(first) >= 0)
    assert set(first.tolist()).issubset(set(values.tolist()))


def test_select_group_values_changes_sample_by_group_id():
    offsets = np.array([0, 20, 40], dtype=np.int64)
    values = np.concatenate(
        [
            np.arange(100, 120, dtype=np.int64),
            np.arange(100, 120, dtype=np.int64),
        ]
    )

    group_zero = select_group_values(
        offsets,
        values,
        0,
        max_values_per_group=8,
        seed=123,
    )
    group_one = select_group_values(
        offsets,
        values,
        1,
        max_values_per_group=8,
        seed=123,
    )

    assert not np.array_equal(group_zero, group_one)


def test_select_group_values_zero_cap_preserves_current_behavior():
    offsets = np.array([0, 12], dtype=np.int64)
    values = np.arange(12, dtype=np.int64)

    selected = select_group_values(
        offsets,
        values,
        0,
        max_values_per_group=0,
        seed=123,
    )

    assert np.array_equal(selected, values)


def test_cap_group_counts_applies_optional_cap():
    counts = np.array([3, 20, 50], dtype=np.int32)

    assert np.array_equal(cap_group_counts(counts, 0), counts)
    assert np.array_equal(cap_group_counts(counts, 10), np.array([3, 10, 10], dtype=np.int32))
