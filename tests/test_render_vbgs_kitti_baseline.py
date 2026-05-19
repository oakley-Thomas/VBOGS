import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import scripts.render_vbgs_kitti_baseline as render_vbgs


def write_colmap_fixture(root, drive="drive_sync"):
    dataset = root / drive
    image_dir = dataset / "images"
    sparse_dir = dataset / "sparse" / "0"
    image_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)

    for frame_id, color in [(1, (255, 0, 0)), (2, (0, 255, 0)), (4, (0, 0, 255))]:
        image = Image.new("RGB", (4, 3), color)
        image.save(image_dir / f"{frame_id:010d}.png")

    (sparse_dir / "cameras.txt").write_text(
        "\n".join(
            [
                "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
                "1 PINHOLE 4 3 2.0 3.0 2.0 1.5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (sparse_dir / "images.txt").write_text(
        "\n".join(
            [
                "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
                "1 1 0 0 0 0 0 0 1 0000000001.png",
                "0.0 0.0 -1",
                "2 1 0 0 0 1 2 3 1 0000000002.png",
                "0.0 0.0 -1",
                "3 1 0 0 0 4 5 6 1 0000000004.png",
                "0.0 0.0 -1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dataset


def test_parse_args_and_default_paths():
    args = render_vbgs.parse_args(["--drive", "drive_sync"])

    model_path, dataset_path, output_dir = render_vbgs.resolve_paths(args)

    assert model_path == (
        render_vbgs.REPO_ROOT / "outputs" / "vbgs_baseline" / "drive_sync" / "model_final.json"
    ).resolve()
    assert dataset_path == (render_vbgs.DEFAULT_COLMAP_ROOT / "drive_sync").resolve()
    assert output_dir == (
        render_vbgs.REPO_ROOT / "outputs" / "vbgs_baseline" / "drive_sync" / "renders"
    ).resolve()


def test_load_kitti_colmap_cameras_reads_intrinsics_and_extrinsics(tmp_path):
    dataset = write_colmap_fixture(tmp_path)

    cameras = render_vbgs.load_kitti_colmap_cameras(dataset)

    assert [camera.frame_id for camera in cameras] == [1, 2, 4]
    assert cameras[0].width == 4
    assert cameras[0].height == 3
    assert cameras[0].CX == 2.0
    assert cameras[0].CY == 1.5
    np.testing.assert_allclose(cameras[0].R, np.eye(3))
    np.testing.assert_allclose(cameras[1].T, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cameras[0].FovX, 2.0 * np.arctan(4.0 / 4.0))
    np.testing.assert_allclose(cameras[0].FovY, 2.0 * np.arctan(3.0 / 6.0))


def test_select_cameras_applies_frame_ids_stride_and_max_views(tmp_path):
    dataset = write_colmap_fixture(tmp_path)
    cameras = render_vbgs.load_kitti_colmap_cameras(dataset)

    selected = render_vbgs.select_cameras(
        cameras,
        frame_ids={1, 2, 4},
        every_n=2,
        max_views=1,
    )

    assert [camera.frame_id for camera in selected] == [1]


def test_load_normalized_model_colors_rescales_raw_kitti_rgb(tmp_path):
    model_path = tmp_path / "model_final.json"
    model_path.write_text(
        json.dumps(
            {
                "mu": [
                    [0, 0, 0, 128, 64, 255],
                    [1, 2, 3, 0, 255, 32],
                ],
                "si": [
                    np.eye(6).tolist(),
                    np.eye(6).tolist(),
                ],
                "alpha": [],
            }
        ),
        encoding="utf-8",
    )

    colors, info = render_vbgs.load_normalized_model_colors(model_path)

    assert info.scale == 255.0
    assert info.source_count == 2
    assert info.renderable_count == 2
    np.testing.assert_allclose(colors[0], [128 / 255, 64 / 255, 1.0])
    np.testing.assert_allclose(colors[1], [0.0, 1.0, 32 / 255])


def test_load_normalized_model_colors_applies_renderable_covariance_mask(tmp_path):
    model_path = tmp_path / "model_final.json"
    bad_covariance = np.eye(6, dtype=np.float32)
    bad_covariance[0, 0] = -1.0
    model_path.write_text(
        json.dumps(
            {
                "mu": [
                    [0, 0, 0, 255, 0, 0],
                    [1, 2, 3, 0, 255, 0],
                    [4, 5, 6, 0, 0, 255],
                ],
                "si": [
                    np.eye(6).tolist(),
                    bad_covariance.tolist(),
                    np.eye(6).tolist(),
                ],
                "alpha": [],
            }
        ),
        encoding="utf-8",
    )

    colors, info = render_vbgs.load_normalized_model_colors(model_path, renderable_count=2)

    assert info.source_count == 3
    assert info.renderable_count == 2
    np.testing.assert_allclose(colors, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_sanitize_model_scales_clamps_pathological_values():
    torch = pytest.importorskip("torch")

    model = SimpleNamespace(
        _scaling=torch.tensor(
            [
                [0.0, 10.0, float("nan")],
                [1.0, 2.0, float("inf")],
                [0.2, 0.3, 0.4],
            ],
            dtype=torch.float32,
        )
    )

    info = render_vbgs.sanitize_model_scales(model, min_scale=0.1, max_scale=5.0)

    assert info.min_scale == 0.1
    assert info.max_scale == 5.0
    assert info.clamped_component_count == 2
    assert float(model._scaling.min()) >= 0.1
    assert float(model._scaling.max()) <= 5.0


def test_render_views_uses_fake_backend_without_cuda(tmp_path, monkeypatch):
    drive = "drive_sync"
    dataset = write_colmap_fixture(tmp_path, drive=drive)
    model_path = tmp_path / "model_final.json"
    model_path.write_text('{"mu": [], "si": [], "alpha": []}', encoding="utf-8")
    output_dir = tmp_path / "renders"

    calls = []

    def fake_model_loader(backend, path, *, device):
        calls.append(("model", str(path), device))
        return {"path": str(path), "device": device}, render_vbgs.ModelColorInfo(
            scale=255.0,
            raw_min=[0.0, 0.0, 0.0],
            raw_max=[255.0, 255.0, 255.0],
            source_count=3,
            renderable_count=2,
        )

    fake_backend = SimpleNamespace()

    def fake_render_camera_image(backend, model, camera, *, device, background, scale):
        calls.append(("render", camera.frame_id, device, background, scale))
        value = 0.25 if camera.frame_id == 1 else 0.75
        return np.full((3, 4, 3), value, dtype=np.float32)

    def fake_sanitize_model_scales(model, *, min_scale, max_scale):
        calls.append(("scales", min_scale, max_scale))
        return render_vbgs.ModelScaleInfo(
            min_scale=min_scale,
            max_scale=max_scale,
            raw_min=[0.0, 0.0, 0.0],
            raw_max=[10.0, 10.0, 10.0],
            sanitized_min=[0.0001, 0.0001, 0.0001],
            sanitized_max=[5.0, 5.0, 5.0],
            clamped_component_count=1,
        )

    monkeypatch.setattr(render_vbgs, "load_render_backend", lambda: fake_backend)
    monkeypatch.setattr(render_vbgs, "load_vbgs_model_for_render", fake_model_loader)
    monkeypatch.setattr(render_vbgs, "sanitize_model_scales", fake_sanitize_model_scales)
    monkeypatch.setattr(render_vbgs, "render_camera_image", fake_render_camera_image)

    args = render_vbgs.parse_args(
        [
            "--drive",
            drive,
            "--model",
            str(model_path),
            "--dataset-path",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--max-views",
            "2",
            "--device",
            "cuda:0",
            "--background",
            "0.5",
            "--scale",
            "2.0",
        ]
    )
    metadata = render_vbgs.render_views(args)

    assert metadata["view_count"] == 2
    assert metadata["camera_count"] == 3
    assert (output_dir / "predicted" / "00000_0000000001.png").exists()
    assert (output_dir / "side_by_side" / "00000_0000000001.png").exists()
    assert (output_dir / "render_metadata.json").exists()
    saved = json.loads((output_dir / "render_metadata.json").read_text(encoding="utf-8"))
    assert saved["views"][0]["frame_id"] == 1
    assert saved["model_color_scale"] == 255.0
    assert saved["model_color_source_count"] == 3
    assert saved["model_color_renderable_count"] == 2
    assert saved["model_scale_clamped_component_count"] == 1
    assert saved["model_scale_sanitized_max"] == [5.0, 5.0, 5.0]
    assert calls[0] == ("model", str(model_path.resolve()), "cuda:0")
    assert calls[1] == ("scales", 0.0001, 5.0)
    assert calls[2] == ("render", 1, "cuda:0", 0.5, 2.0)
