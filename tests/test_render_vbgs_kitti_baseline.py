import json
from types import SimpleNamespace

import numpy as np
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


def test_render_views_uses_fake_backend_without_cuda(tmp_path, monkeypatch):
    drive = "drive_sync"
    dataset = write_colmap_fixture(tmp_path, drive=drive)
    model_path = tmp_path / "model_final.json"
    model_path.write_text('{"mu": [], "si": [], "alpha": []}', encoding="utf-8")
    output_dir = tmp_path / "renders"

    calls = []

    def fake_model_loader(path, device):
        calls.append(("model", str(path), device))
        return {"path": str(path), "device": device}

    fake_backend = SimpleNamespace(vbgs_model_to_splat=fake_model_loader)

    def fake_render_camera_image(backend, model, camera, *, device, background, scale):
        calls.append(("render", camera.frame_id, device, background, scale))
        value = 0.25 if camera.frame_id == 1 else 0.75
        return np.full((3, 4, 3), value, dtype=np.float32)

    monkeypatch.setattr(render_vbgs, "load_render_backend", lambda: fake_backend)
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
    assert calls[0] == ("model", str(model_path.resolve()), "cuda:0")
    assert calls[1] == ("render", 1, "cuda:0", 0.5, 2.0)
