from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.view_octree_anygs import parse_args
from vbogs.viewer.camera import ViewerCamera, coerce_c2w
from vbogs.viewer.rendering import (
    RenderedFrame,
    compose_layer,
    normalized_uncertainty,
    tensor_to_jpeg,
    validate_uncertainty_array,
)
from vbogs.viewer.server import LatestRequestBuffer, create_app


def test_viewer_cli_defaults_are_docker_friendly():
    args = parse_args([])

    assert args.host == "0.0.0.0"
    assert args.port == 8070
    assert args.resolution == 4
    assert args.camera_source == "test"
    assert args.camera_index == 0
    assert args.rgb_only is False


def test_validate_uncertainty_array_rejects_length_mismatch():
    with pytest.raises(ValueError, match="scene has 4 anchors"):
        validate_uncertainty_array(np.zeros((3,), dtype=np.float32), anchor_count=4)


def test_validate_uncertainty_array_rejects_non_vector():
    with pytest.raises(ValueError, match="1D uncertainty"):
        validate_uncertainty_array(np.zeros((1, 3), dtype=np.float32), anchor_count=3)


def test_coerce_c2w_requires_4x4_matrix():
    with pytest.raises(ValueError, match=r"\(4, 4\)"):
        coerce_c2w(np.eye(3, dtype=np.float32))


def test_viewer_camera_wraps_custom_pose_on_cpu():
    source = SimpleNamespace(
        uid=7,
        image_name="frame",
        image_path="",
        resolution_scale=1.0,
        image_width=8,
        image_height=6,
        fx=4.0,
        fy=5.0,
        cx=4.0,
        cy=3.0,
        znear=0.01,
        zfar=100.0,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    camera = ViewerCamera.from_source(source, c2w=c2w, device="cpu")

    assert camera.uid == 7
    assert camera.image_width == 8
    assert camera.image_height == 6
    assert camera.camera_center.detach().cpu().tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_normalized_uncertainty_divides_by_alpha_only_where_visible():
    unc = torch.tensor([[2.0, 5.0], [0.0, 8.0]])
    alpha = torch.tensor([[0.5, 0.0], [0.0, 2.0]])

    display = normalized_uncertainty(unc, alpha)

    assert torch.allclose(display, torch.tensor([[4.0, 0.0], [0.0, 4.0]]))


def test_compose_side_by_side_has_double_width():
    rgb = torch.ones((3, 2, 3), dtype=torch.float32)
    unc = torch.ones((2, 3), dtype=torch.float32)
    alpha = torch.ones((2, 3), dtype=torch.float32)

    image = compose_layer(
        mode="side_by_side",
        rgb=rgb,
        unc_image=unc,
        alpha_image=alpha,
        vmin=0.0,
        vmax=1.0,
        colormap_name="turbo",
    )

    assert tuple(image.shape) == (3, 2, 6)


def test_tensor_to_jpeg_returns_jpeg_bytes():
    image = torch.zeros((3, 4, 5), dtype=torch.float32)

    payload = tensor_to_jpeg(image, quality=80)

    assert payload[:2] == b"\xff\xd8"


def test_latest_request_buffer_keeps_only_newest():
    buffer = LatestRequestBuffer()

    buffer.replace({"request_id": "old"})
    buffer.replace({"request_id": "new"})

    assert buffer.take_latest() == {"request_id": "new"}
    assert buffer.is_empty()


def test_fastapi_routes_and_websocket_with_fake_session():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class FakeSession:
        def __init__(self) -> None:
            self.requests = []

        def metadata(self):
            return {
                "drive": "drive",
                "anchor_count": 1,
                "render_modes": ["rgb"],
                "default_camera_id": "test:0",
                "rgb_only": True,
                "max_fps": 20,
                "jpeg_quality": 80,
                "model_path": "/tmp/model",
            }

        def camera_payload(self):
            return {
                "default_camera_id": "test:0",
                "cameras": {
                    "train": [],
                    "test": [
                        {
                            "id": "test:0",
                            "source": "test",
                            "index": 0,
                            "image_name": "frame",
                            "width": 4,
                            "height": 3,
                            "fx": 1.0,
                            "fy": 1.0,
                            "cx": 2.0,
                            "cy": 1.5,
                            "c2w": np.eye(4, dtype=float).tolist(),
                        }
                    ],
                },
            }

        def render_request(self, payload):
            self.requests.append(payload)
            return RenderedFrame(
                metadata={
                    "request_id": payload.get("request_id"),
                    "mode": payload.get("layer"),
                    "width": 1,
                    "height": 1,
                    "elapsed_ms": 1.0,
                    "jpeg_quality": 80,
                },
                jpeg=b"\xff\xd8fake",
            )

    session = FakeSession()
    client = TestClient(create_app(session))

    assert client.get("/api/metadata").json()["drive"] == "drive"
    assert client.get("/api/cameras").json()["default_camera_id"] == "test:0"
    assert client.get("/").status_code == 200

    with client.websocket_connect("/ws/render") as websocket:
        websocket.send_json({"request_id": "r1", "layer": "rgb"})
        assert websocket.receive_json()["request_id"] == "r1"
        assert websocket.receive_bytes() == b"\xff\xd8fake"

    assert session.requests == [{"request_id": "r1", "layer": "rgb"}]
