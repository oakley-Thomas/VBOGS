import numpy as np

from scripts.export_uncertainty_map import (
    PLY_DTYPE,
    choose_color_scale,
    export_uncertainty_map,
    uncertainty_to_rgb,
)


def read_binary_ply(path):
    data = path.read_bytes()
    marker = b"end_header\n"
    header_end = data.index(marker) + len(marker)
    header = data[:header_end].decode("ascii")
    rows = np.frombuffer(data[header_end:], dtype=PLY_DTYPE)
    return header, rows


def test_uncertainty_to_rgb_maps_low_blue_and_high_red():
    rgb = uncertainty_to_rgb(
        np.array([0.0, 5.0, 10.0], dtype=np.float32),
        np.array([True, True, True]),
        vmin=0.0,
        vmax=10.0,
    )

    assert rgb[0].tolist() == [0, 0, 255]
    assert rgb[2].tolist() == [255, 0, 0]


def test_uncertainty_to_rgb_clips_outside_scale():
    rgb = uncertainty_to_rgb(
        np.array([-10.0, 0.0, 10.0, 20.0], dtype=np.float32),
        np.array([True, True, True, True]),
        vmin=0.0,
        vmax=10.0,
    )

    assert rgb[0].tolist() == [0, 0, 255]
    assert rgb[1].tolist() == [0, 0, 255]
    assert rgb[2].tolist() == [255, 0, 0]
    assert rgb[3].tolist() == [255, 0, 0]


def test_unobserved_anchor_is_red_even_when_numeric_value_equals_observed_max():
    rgb = uncertainty_to_rgb(
        np.array([0.0, 10.0, 10.0], dtype=np.float32),
        np.array([True, True, False]),
        vmin=0.0,
        vmax=20.0,
    )

    assert rgb[1].tolist() == [128, 0, 128]
    assert rgb[2].tolist() == [255, 0, 0]


def test_choose_color_scale_uses_observed_percentiles():
    uncertainty = np.array([0.0, 10.0, 20.0, 1000.0], dtype=np.float32)
    is_observed = np.array([True, True, True, False])

    vmin, vmax, source = choose_color_scale(
        uncertainty,
        is_observed,
        vmin=None,
        vmax=None,
        percentile_low=0.0,
        percentile_high=100.0,
    )

    assert vmin == 0.0
    assert vmax == 20.0
    assert source == "observed_percentiles"


def test_export_uncertainty_map_writes_all_and_lod_plys(tmp_path):
    bucket_root = tmp_path / "m4" / "drive"
    bucket_root.mkdir(parents=True)
    output_dir = tmp_path / "viz"
    uncertainty_path = bucket_root / "U.npy"
    posterior_path = bucket_root / "anchor_posterior.npz"

    np.savez_compressed(
        bucket_root / "pts_by_anchor.npz",
        anchor_xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        anchor_level=np.array([0, 1, 1, 2], dtype=np.int16),
        point_counts=np.array([3, 0, 5, 7], dtype=np.int32),
        voxel_size=np.array(8.0, dtype=np.float32),
        fork=np.array(2, dtype=np.int16),
        levels=np.array(3, dtype=np.int16),
    )
    np.save(uncertainty_path, np.array([0.0, 3.0, 1.0, 2.0], dtype=np.float32))
    np.savez_compressed(
        posterior_path,
        is_observed=np.array([True, False, True, True]),
    )

    metadata = export_uncertainty_map(
        bucket_root=bucket_root,
        uncertainty_path=uncertainty_path,
        posterior_path=posterior_path,
        output_dir=output_dir,
        vmin=0.0,
        vmax=3.0,
        percentile_low=2.0,
        percentile_high=98.0,
        observed_only=False,
        split_levels=True,
    )

    all_path = output_dir / "anchors_uncertainty_all.ply"
    lod_01_path = output_dir / "anchors_uncertainty_lod_01.ply"
    assert all_path.exists()
    assert lod_01_path.exists()
    assert (output_dir / "uncertainty_map_metadata.json").exists()
    assert metadata["exported_anchor_count"] == 4
    assert metadata["level_counts"] == {"0": 1, "1": 2, "2": 1}

    header, rows = read_binary_ply(all_path)
    assert "property float uncertainty" in header
    assert "property int anchor_id" in header
    assert "property short level" in header
    assert "property uchar is_observed" in header
    assert rows.shape == (4,)
    assert rows["anchor_id"].tolist() == [0, 1, 2, 3]
    assert rows["red"][1] == 255
    assert rows["green"][1] == 0
    assert rows["blue"][1] == 0
    assert rows["cell_size"].tolist() == [8.0, 4.0, 4.0, 2.0]

    _, lod_01_rows = read_binary_ply(lod_01_path)
    assert lod_01_rows.shape == (2,)
    assert lod_01_rows["level"].tolist() == [1, 1]
    assert lod_01_rows["anchor_id"].tolist() == [1, 2]
