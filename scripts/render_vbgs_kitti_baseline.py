#!/usr/bin/env python3

"""Render a base VBGS KITTI baseline model with prepared KITTI cameras."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_COLMAP_ROOT = Path("/data/COLMAP")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "vbgs_baseline"


@dataclass(frozen=True)
class PinholeCamera:
    camera_id: int
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


@dataclass(frozen=True)
class KittiCameraInfo:
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    CX: float
    CY: float
    image: Image.Image
    mask: Any | None
    depth: Any | None
    depth_params: Any | None
    image_path: str
    image_name: str
    width: int
    height: int
    frame_id: int | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive",
        default="2013_05_28_drive_0007_sync",
        help="KITTI-360 drive id used to resolve default paths.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Base VBGS model JSON. Defaults to outputs/vbgs_baseline/<drive>/model_final.json.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Prepared KITTI COLMAP dataset. Defaults to /data/COLMAP/<drive>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Render output directory. Defaults to outputs/vbgs_baseline/<drive>/renders.",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=0,
        help="Optional cap after filtering. 0 renders every selected view.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Render every Nth selected camera.",
    )
    parser.add_argument(
        "--frame-ids",
        nargs="*",
        default=None,
        help="Optional frame ids to render. Accepts spaces and/or comma-separated lists.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device for rendering.")
    parser.add_argument(
        "--background",
        type=float,
        default=0.0,
        help="Scalar RGB background value passed to the Graphdeco renderer.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.41,
        help="Scale modifier forwarded to the Graphdeco renderer.",
    )
    parser.add_argument(
        "--no-side-by-side",
        action="store_true",
        help="Only write predicted renders, not ground-truth/prediction pairs.",
    )
    return parser.parse_args(argv)


def default_model_path(drive: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / drive / "model_final.json"


def default_dataset_path(drive: str) -> Path:
    return DEFAULT_COLMAP_ROOT / drive


def default_output_dir(drive: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / drive / "renders"


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    model_path = (args.model or default_model_path(args.drive)).resolve()
    dataset_path = (args.dataset_path or default_dataset_path(args.drive)).resolve()
    output_dir = (args.output_dir or default_output_dir(args.drive)).resolve()
    return model_path, dataset_path, output_dir


def focal2fov(focal: float, pixels: int) -> float:
    if focal <= 0.0:
        raise ValueError(f"Focal length must be positive, got {focal}")
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    if qvec.shape != (4,):
        raise ValueError(f"Expected qvec shape (4,), got {qvec.shape}")
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ],
        dtype=np.float64,
    )


def data_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def read_colmap_cameras(path: Path) -> dict[int, PinholeCamera]:
    cameras: dict[int, PinholeCamera] = {}
    for line in data_lines(path):
        tokens = line.split()
        if len(tokens) < 5:
            raise ValueError(f"Malformed COLMAP camera line in {path}: {line}")
        camera_id = int(tokens[0])
        model = tokens[1]
        width = int(tokens[2])
        height = int(tokens[3])
        params = [float(value) for value in tokens[4:]]
        if model == "PINHOLE":
            if len(params) != 4:
                raise ValueError(f"PINHOLE camera {camera_id} needs 4 params")
            fx, fy, cx, cy = params
        elif model == "SIMPLE_PINHOLE":
            if len(params) != 3:
                raise ValueError(f"SIMPLE_PINHOLE camera {camera_id} needs 3 params")
            fx = fy = params[0]
            cx, cy = params[1:]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {model}")
        cameras[camera_id] = PinholeCamera(
            camera_id=camera_id,
            model=model,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
    if not cameras:
        raise ValueError(f"No cameras found in {path}")
    return cameras


def read_colmap_images(path: Path) -> list[ColmapImage]:
    if not path.exists():
        raise FileNotFoundError(path)
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("#"):
                lines.append(line)

    images: list[ColmapImage] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue
        tokens = line.split()
        if len(tokens) < 10:
            raise ValueError(f"Malformed COLMAP image line in {path}: {line}")
        name = " ".join(tokens[9:])
        images.append(
            ColmapImage(
                image_id=int(tokens[0]),
                qvec=np.asarray([float(value) for value in tokens[1:5]], dtype=np.float64),
                tvec=np.asarray([float(value) for value in tokens[5:8]], dtype=np.float64),
                camera_id=int(tokens[8]),
                name=name,
            )
        )
        # COLMAP text images are stored as image metadata followed by a POINTS2D line.
        index += 2
    if not images:
        raise ValueError(f"No images found in {path}")
    return images


def image_frame_id(image_name: str) -> int | None:
    stem = Path(image_name).stem
    try:
        return int(stem)
    except ValueError:
        return None


def load_kitti_colmap_cameras(dataset_path: Path) -> list[KittiCameraInfo]:
    sparse_dir = dataset_path / "sparse" / "0"
    cameras = read_colmap_cameras(sparse_dir / "cameras.txt")
    images = read_colmap_images(sparse_dir / "images.txt")
    image_root = dataset_path / "images"

    camera_infos: list[KittiCameraInfo] = []
    for item in sorted(images, key=lambda image: image.name):
        calibration = cameras.get(item.camera_id)
        if calibration is None:
            raise ValueError(f"Image {item.name} references missing camera {item.camera_id}")

        image_path = image_root / item.name
        if not image_path.exists():
            raise FileNotFoundError(f"Prepared image not found: {image_path}")
        with Image.open(image_path) as handle:
            image = handle.convert("RGB").copy()

        R = qvec2rotmat(item.qvec).T
        camera_infos.append(
            KittiCameraInfo(
                uid=item.image_id,
                R=R,
                T=item.tvec.astype(np.float64),
                FovY=focal2fov(calibration.fy, calibration.height),
                FovX=focal2fov(calibration.fx, calibration.width),
                CX=calibration.cx,
                CY=calibration.cy,
                image=image,
                mask=None,
                depth=None,
                depth_params=None,
                image_path=str(image_path),
                image_name=Path(item.name).stem,
                width=image.size[0],
                height=image.size[1],
                frame_id=image_frame_id(item.name),
            )
        )
    return camera_infos


def parse_frame_ids(raw_values: Sequence[str] | None) -> set[int] | None:
    if raw_values is None:
        return None
    frame_ids: set[int] = set()
    for raw_value in raw_values:
        for token in raw_value.split(","):
            token = token.strip()
            if token:
                frame_ids.add(int(token))
    return frame_ids


def select_cameras(
    cameras: Sequence[KittiCameraInfo],
    *,
    frame_ids: set[int] | None,
    every_n: int,
    max_views: int,
) -> list[KittiCameraInfo]:
    if every_n <= 0:
        raise ValueError("--every-n must be positive")
    if max_views < 0:
        raise ValueError("--max-views must be non-negative")

    selected = list(cameras)
    if frame_ids is not None:
        selected = [camera for camera in selected if camera.frame_id in frame_ids]
    selected = selected[::every_n]
    if max_views:
        selected = selected[:max_views]
    if not selected:
        raise ValueError("No cameras selected for rendering")
    return selected


def load_render_backend() -> Any:
    try:
        from vbgs.render import volume as backend
    except Exception as exc:  # pragma: no cover - exercised in render env
        raise RuntimeError(
            "Base VBGS rendering requires the dedicated render environment with "
            "vbgs and Graphdeco gaussian-splatting installed."
        ) from exc
    return backend


def configure_backend_args(backend: Any, *, device: str) -> Any:
    cargs = getattr(backend, "cargs", None)
    if cargs is None:
        raise RuntimeError("VBGS render backend does not expose `cargs`")
    cargs.data_device = device
    if not hasattr(cargs, "data_format"):
        cargs.data_format = "colmap"
    return cargs


def load_backend_camera(backend: Any, camera: KittiCameraInfo, *, device: str) -> Any:
    cargs = configure_backend_args(backend, device=device)
    load_cam = backend.loadCam
    parameter_count = len(inspect.signature(load_cam).parameters)
    if parameter_count >= 5:
        return load_cam(cargs, 0, camera, 1.0, None)
    return load_cam(cargs, 0, camera, 1.0)


def render_camera_image(
    backend: Any,
    model: Any,
    camera: KittiCameraInfo,
    *,
    device: str,
    background: float,
    scale: float,
) -> np.ndarray:
    import torch

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)

    render_camera = load_backend_camera(backend, camera, device=device)
    background_tensor = float(background) * torch.ones(3, device=device)
    rendered = backend.render_cuda(
        render_camera,
        model,
        backend.pipe,
        background_tensor,
        scale,
    )["render"]
    image = rendered.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def array_to_image(values: np.ndarray) -> Image.Image:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] < 3:
        raise ValueError(f"Expected rendered image shape (H, W, C>=3), got {values.shape}")
    values = np.nan_to_num(values[..., :3], nan=0.0, posinf=1.0, neginf=0.0)
    pixels = (np.clip(values, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def save_side_by_side(reference: Image.Image, prediction: Image.Image, path: Path) -> None:
    reference = reference.convert("RGB")
    if reference.size != prediction.size:
        reference = reference.resize(prediction.size, Image.BILINEAR)
    width, height = prediction.size
    canvas = Image.new("RGB", (width * 2, height))
    canvas.paste(reference, (0, 0))
    canvas.paste(prediction, (width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def render_views(args: argparse.Namespace) -> dict[str, Any]:
    model_path, dataset_path, output_dir = resolve_paths(args)
    if not model_path.exists():
        raise FileNotFoundError(f"Base VBGS model not found: {model_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prepared KITTI COLMAP dataset not found: {dataset_path}")

    cameras = load_kitti_colmap_cameras(dataset_path)
    selected_cameras = select_cameras(
        cameras,
        frame_ids=parse_frame_ids(args.frame_ids),
        every_n=args.every_n,
        max_views=args.max_views,
    )

    output_predicted = output_dir / "predicted"
    output_side_by_side = output_dir / "side_by_side"
    output_predicted.mkdir(parents=True, exist_ok=True)
    if not args.no_side_by_side:
        output_side_by_side.mkdir(parents=True, exist_ok=True)

    backend = load_render_backend()
    model = backend.vbgs_model_to_splat(model_path, device=args.device)

    rows: list[dict[str, Any]] = []
    for index, camera in enumerate(selected_cameras):
        start = time.time()
        rendered = render_camera_image(
            backend,
            model,
            camera,
            device=args.device,
            background=args.background,
            scale=args.scale,
        )
        prediction = array_to_image(rendered)

        stem = f"{index:05d}_{camera.image_name}"
        predicted_path = output_predicted / f"{stem}.png"
        predicted_path.parent.mkdir(parents=True, exist_ok=True)
        prediction.save(predicted_path)

        side_by_side_path: Path | None = None
        if not args.no_side_by_side:
            side_by_side_path = output_side_by_side / f"{stem}.png"
            save_side_by_side(camera.image, prediction, side_by_side_path)

        rows.append(
            {
                "index": index,
                "frame_id": camera.frame_id,
                "image_name": camera.image_name,
                "source_image": camera.image_path,
                "predicted": str(predicted_path),
                "side_by_side": str(side_by_side_path) if side_by_side_path else None,
                "width": int(prediction.size[0]),
                "height": int(prediction.size[1]),
                "seconds": time.time() - start,
            }
        )

    metadata = {
        "drive": args.drive,
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "device": args.device,
        "background": float(args.background),
        "scale": float(args.scale),
        "max_views": int(args.max_views),
        "every_n": int(args.every_n),
        "frame_ids": sorted(parse_frame_ids(args.frame_ids) or []),
        "side_by_side": not args.no_side_by_side,
        "camera_count": len(cameras),
        "view_count": len(rows),
        "views": rows,
    }
    metadata_path = output_dir / "render_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = render_views(args)
    print(f"Wrote {metadata['view_count']} base VBGS renders under {metadata['output_dir']}")


if __name__ == "__main__":
    main()
