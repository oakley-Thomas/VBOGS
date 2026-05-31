from __future__ import annotations

import base64

import pytest

from scripts import capture_viewer_api_views as capture


class FakeViewerApiClient:
    def __init__(self) -> None:
        self.base_url = "http://viewer.test"
        self.posts: list[tuple[str, dict]] = []

    def get_json(self, path: str) -> dict:
        if path == "/api/metadata":
            return {
                "drive": "drive_sync",
                "rgb_only": False,
            }
        if path == "/api/cameras":
            return {
                "cameras": {
                    "train": [
                        {"id": "train:0", "image_name": "frame_0000000000.png"},
                        {"id": "train:1", "image_name": "frame_0000000001.png"},
                        {"id": "train:2", "image_name": "frame_0000000002.png"},
                    ],
                    "test": [{"id": "test:0", "image_name": "heldout.png"}],
                }
            }
        raise AssertionError(f"Unexpected GET {path}")

    def post_json(self, path: str, payload: dict) -> dict:
        self.posts.append((path, payload))
        if path == "/api/rendered-anchors":
            return {
                "request_id": payload["request_id"],
                "camera_id": payload["camera_id"],
                "anchor_count_rendered": 2,
                "anchor_count_returned": 2,
                "truncated": False,
                "total_anchor_uncertainty": 5.0,
                "uncertainty_image_sum": 12.0,
                "alpha_sum": 3.0,
                "alpha_normalized_uncertainty": 4.0,
                "anchors": [
                    {"anchor_id": 1, "xyz": [0.0, 0.0, 0.0], "uncertainty": 2.0},
                    {"anchor_id": 2, "xyz": [1.0, 0.0, 0.0], "uncertainty": 3.0},
                ],
            }
        return {
            "metadata": {
                "request_id": payload["request_id"],
                "camera_id": payload["camera_id"],
                "mode": payload["layer"],
                "width": 4,
                "height": 3,
            },
            "jpeg_base64": base64.b64encode(b"\xff\xd8fake-jpeg").decode("ascii"),
        }


def test_capture_views_saves_first_training_renders(tmp_path):
    client = FakeViewerApiClient()

    manifest = capture.capture_views(
        client,
        source="train",
        count=2,
        layer="rgb",
        quality=81,
        output_dir=tmp_path,
    )

    assert manifest["captured_count"] == 2
    assert [post[0] for post in client.posts] == [
        "/api/render",
        "/api/rendered-anchors",
        "/api/render",
        "/api/rendered-anchors",
    ]
    render_posts = [post for post in client.posts if post[0] == "/api/render"]
    anchor_posts = [post for post in client.posts if post[0] == "/api/rendered-anchors"]
    assert [post[1]["camera_id"] for post in render_posts] == ["train:0", "train:1"]
    assert [post[1]["layer"] for post in render_posts] == ["rgb", "rgb"]
    assert [post[1]["quality"] for post in render_posts] == [81, 81]
    assert [post[1]["camera_id"] for post in anchor_posts] == ["train:0", "train:1"]
    assert (tmp_path / "capture_manifest.json").exists()

    images = sorted(tmp_path.glob("*.jpg"))
    sidecars = sorted(
        path
        for path in tmp_path.glob("*.json")
        if not path.name.endswith("_anchors.json") and path.name != "capture_manifest.json"
    )
    anchor_files = sorted(tmp_path.glob("*_anchors.json"))
    assert len(images) == 2
    assert len(sidecars) == 2
    assert len(anchor_files) == 2
    assert images[0].read_bytes() == b"\xff\xd8fake-jpeg"
    assert manifest["captures"][0]["rendered_anchors"]["total_anchor_uncertainty"] == 5.0


def test_capture_views_can_skip_rendered_anchor_endpoint(tmp_path):
    client = FakeViewerApiClient()

    manifest = capture.capture_views(
        client,
        source="train",
        count=1,
        layer="rgb",
        quality=81,
        output_dir=tmp_path,
        include_rendered_anchors=False,
    )

    assert [post[0] for post in client.posts] == ["/api/render"]
    assert manifest["captures"][0]["anchors_path"] is None


def test_capture_views_forwards_max_rendered_anchors(tmp_path):
    client = FakeViewerApiClient()

    capture.capture_views(
        client,
        source="train",
        count=1,
        layer="rgb",
        quality=81,
        output_dir=tmp_path,
        max_rendered_anchors=7,
    )

    anchor_post = [post for post in client.posts if post[0] == "/api/rendered-anchors"][0]
    assert anchor_post[1]["max_anchors"] == 7


def test_select_camera_summaries_rejects_missing_source():
    with pytest.raises(ValueError, match="source 'train'"):
        capture.select_camera_summaries({"cameras": {"test": []}}, source="train", count=1)


def test_parse_args_defaults_to_first_training_side_by_side():
    args = capture.parse_args([])

    assert args.base_url == "http://localhost:8070"
    assert args.source == "train"
    assert args.count == 5
    assert args.layer == "side_by_side"
    assert args.skip_rendered_anchors is False
