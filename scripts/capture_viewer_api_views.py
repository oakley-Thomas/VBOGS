#!/usr/bin/env python3

"""Capture viewer renders for the first cameras exposed by the realtime API."""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vbogs.io import save_json

DEFAULT_BASE_URL = "http://localhost:8070"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "viewer_api_captures"
RENDER_LAYERS = ("rgb", "uncertainty", "alpha", "side_by_side")


class ViewerApiClient:
    """Tiny stdlib client for the realtime viewer REST API."""

    def __init__(self, base_url: str, *, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def get_json(self, path: str) -> dict[str, Any]:
        return self._request_json("GET", path)

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json("POST", path, payload=payload)

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach viewer API at {self.base_url}: {exc}") from exc
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError(f"{method} {path} returned non-object JSON")
        return decoded


def safe_stem(value: str) -> str:
    stem = Path(str(value)).stem
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return cleaned or "camera"


def resolve_output_dir(output_dir: Path | None, metadata: dict[str, Any], *, source: str, layer: str) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    drive = safe_stem(str(metadata.get("drive", "unknown_drive")))
    return (DEFAULT_OUTPUT_ROOT / drive / source / layer).resolve()


def select_camera_summaries(camera_payload: dict[str, Any], *, source: str, count: int) -> list[dict[str, Any]]:
    cameras_by_source = camera_payload.get("cameras")
    if not isinstance(cameras_by_source, dict):
        raise ValueError("/api/cameras response does not contain a cameras object")
    cameras = cameras_by_source.get(source)
    if not isinstance(cameras, list):
        raise ValueError(f"/api/cameras response does not contain source {source!r}")
    if count <= 0:
        raise ValueError("--count must be positive")
    selected = cameras[:count]
    if not selected:
        raise ValueError(f"No {source} cameras are available from the viewer API")
    return selected


def decode_jpeg(render_response: dict[str, Any]) -> bytes:
    encoded = render_response.get("jpeg_base64")
    if not isinstance(encoded, str):
        raise ValueError("/api/render response does not contain jpeg_base64")
    payload = base64.b64decode(encoded)
    if not payload.startswith(b"\xff\xd8"):
        raise ValueError("/api/render did not return JPEG bytes")
    return payload


def capture_views(
    client: ViewerApiClient,
    *,
    source: str,
    count: int,
    layer: str,
    quality: int,
    output_dir: Path | None,
    include_rendered_anchors: bool = True,
    max_rendered_anchors: int | None = None,
) -> dict[str, Any]:
    if source not in ("train", "test"):
        raise ValueError("--source must be train or test")
    if layer not in RENDER_LAYERS:
        raise ValueError(f"--layer must be one of: {', '.join(RENDER_LAYERS)}")
    quality = max(1, min(95, int(quality)))

    metadata = client.get_json("/api/metadata")
    if metadata.get("rgb_only") and layer != "rgb":
        raise ValueError("Viewer was started with --rgb-only; use --layer rgb")
    if metadata.get("rgb_only") and include_rendered_anchors:
        raise ValueError("Viewer was started with --rgb-only; rendered-anchor capture is unavailable")
    camera_payload = client.get_json("/api/cameras")
    selected = select_camera_summaries(camera_payload, source=source, count=count)
    resolved_output = resolve_output_dir(output_dir, metadata, source=source, layer=layer)
    resolved_output.mkdir(parents=True, exist_ok=True)

    captures: list[dict[str, Any]] = []
    for index, camera in enumerate(selected):
        camera_id = str(camera.get("id", f"{source}:{index}"))
        image_name = safe_stem(str(camera.get("image_name", camera_id)))
        request_id = f"capture-{source}-{index:04d}"
        render_response = client.post_json(
            "/api/render",
            {
                "request_id": request_id,
                "camera_id": camera_id,
                "layer": layer,
                "quality": quality,
            },
        )
        jpeg = decode_jpeg(render_response)
        filename = f"{index:04d}_{safe_stem(camera_id)}_{image_name}_{layer}.jpg"
        image_path = resolved_output / filename
        metadata_path = image_path.with_suffix(".json")
        anchors_path = image_path.with_name(f"{image_path.stem}_anchors.json")
        image_path.write_bytes(jpeg)

        render_metadata = render_response.get("metadata")
        if not isinstance(render_metadata, dict):
            render_metadata = {}
        rendered_anchors = None
        rendered_anchor_summary = None
        if include_rendered_anchors:
            anchor_request = {
                "request_id": f"{request_id}-anchors",
                "camera_id": camera_id,
            }
            if max_rendered_anchors is not None:
                anchor_request["max_anchors"] = max(0, int(max_rendered_anchors))
            rendered_anchors = client.post_json("/api/rendered-anchors", anchor_request)
            save_json(anchors_path, rendered_anchors)
            rendered_anchor_summary = {
                "path": str(anchors_path),
                "anchor_count_rendered": rendered_anchors.get("anchor_count_rendered"),
                "anchor_count_returned": rendered_anchors.get("anchor_count_returned"),
                "truncated": rendered_anchors.get("truncated"),
                "total_anchor_uncertainty": rendered_anchors.get("total_anchor_uncertainty"),
                "uncertainty_image_sum": rendered_anchors.get("uncertainty_image_sum"),
                "alpha_sum": rendered_anchors.get("alpha_sum"),
                "alpha_normalized_uncertainty": rendered_anchors.get("alpha_normalized_uncertainty"),
            }
        capture = {
            "index": index,
            "camera": camera,
            "request": {
                "request_id": request_id,
                "camera_id": camera_id,
                "layer": layer,
                "quality": quality,
            },
            "render_metadata": render_metadata,
            "rendered_anchors": rendered_anchor_summary,
            "image_path": str(image_path),
            "metadata_path": str(metadata_path),
            "anchors_path": str(anchors_path) if rendered_anchors is not None else None,
        }
        save_json(metadata_path, capture)
        captures.append(capture)

    manifest = {
        "base_url": client.base_url,
        "viewer_metadata": metadata,
        "source": source,
        "requested_count": count,
        "captured_count": len(captures),
        "layer": layer,
        "quality": quality,
        "include_rendered_anchors": include_rendered_anchors,
        "max_rendered_anchors": max_rendered_anchors,
        "output_dir": str(resolved_output),
        "captures": captures,
    }
    save_json(resolved_output / "capture_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Viewer API base URL.")
    parser.add_argument("--source", choices=("train", "test"), default="train", help="Camera split to capture.")
    parser.add_argument("--count", type=int, default=5, help="Number of cameras to render from the start of the split.")
    parser.add_argument("--layer", choices=RENDER_LAYERS, default="side_by_side", help="Render layer to request.")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality passed to /api/render.")
    parser.add_argument(
        "--skip-rendered-anchors",
        action="store_true",
        help="Do not call /api/rendered-anchors for each captured view.",
    )
    parser.add_argument(
        "--max-rendered-anchors",
        type=int,
        default=None,
        help="Limit anchor rows returned by /api/rendered-anchors; totals still cover all rendered anchors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/viewer_api_captures/<drive>/<source>/<layer>.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    client = ViewerApiClient(args.base_url, timeout=args.timeout)
    manifest = capture_views(
        client,
        source=args.source,
        count=args.count,
        layer=args.layer,
        quality=args.quality,
        output_dir=args.output_dir,
        include_rendered_anchors=not args.skip_rendered_anchors,
        max_rendered_anchors=args.max_rendered_anchors,
    )
    print(f"Wrote {manifest['captured_count']} {args.source} capture(s) to {manifest['output_dir']}")
    print(f"Wrote {Path(manifest['output_dir']) / 'capture_manifest.json'}")


if __name__ == "__main__":
    main()
