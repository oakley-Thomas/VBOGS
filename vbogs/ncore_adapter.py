"""Small helpers for NVIDIA PhysicalAI AV NCore dataset adapters.

The real NCore dependency is imported lazily by scripts so unit tests can use
duck-typed fake loaders without installing the gated dataset tooling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_CAMERA_IDS = ("camera_front_wide_120fov",)
DEFAULT_CAMERA_DEPTH_PAIR = (
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
)
DEFAULT_LIDAR_ID = "lidar_top_360fov"


@dataclass(frozen=True)
class PinholeCamera:
    camera_id: str
    camera_model_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    camera_model_id: int
    image_name: str
    c2w: np.ndarray


def parse_id_list(values: Sequence[str] | str | None, default: Sequence[str]) -> list[str]:
    if values is None:
        values = default
    if isinstance(values, str):
        values = [values]
    parsed: list[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token and token not in parsed:
                parsed.append(token)
    return parsed or list(default)


def json_ready_matrix(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix).reshape(4, 4)]


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    hom = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1))], axis=1)
    return (transform @ hom.T).T[:, :3].astype(np.float32)


def rotmat_to_qvec(rot: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = np.asarray(rot).reshape(3, 3).flat
    k = np.array(
        [
            [rxx - ryy - rzz, 0.0, 0.0, 0.0],
            [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def _maybe_attr(obj: Any, names: Iterable[str]) -> Any | None:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _camera_matrix(model_parameters: Any) -> np.ndarray | None:
    for name in ("camera_matrix", "intrinsic_matrix", "K", "matrix"):
        value = _maybe_attr(model_parameters, (name,))
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64)
        if arr.size == 9:
            return arr.reshape(3, 3)
    return None


def extract_pinhole_camera(
    *,
    camera_id: str,
    camera_model_id: int,
    model_parameters: Any,
    image_shape: Sequence[int],
) -> PinholeCamera:
    """Extract a practical PINHOLE camera from NCore-style model parameters."""

    height = int(_maybe_attr(model_parameters, ("height", "image_height", "rows")) or image_shape[0])
    width = int(_maybe_attr(model_parameters, ("width", "image_width", "cols")) or image_shape[1])
    resolution = _maybe_attr(model_parameters, ("resolution", "image_resolution"))
    if resolution is not None:
        resolution_arr = np.asarray(resolution).reshape(-1)
        if resolution_arr.size >= 2:
            width = int(resolution_arr[0])
            height = int(resolution_arr[1])

    matrix = _camera_matrix(model_parameters)
    if matrix is not None:
        fx = float(matrix[0, 0])
        fy = float(matrix[1, 1])
        cx = float(matrix[0, 2])
        cy = float(matrix[1, 2])
    else:
        fx = _maybe_attr(model_parameters, ("fx", "focal_x", "focal_length_x"))
        fy = _maybe_attr(model_parameters, ("fy", "focal_y", "focal_length_y"))
        cx = _maybe_attr(model_parameters, ("cx", "principal_point_x", "center_x"))
        cy = _maybe_attr(model_parameters, ("cy", "principal_point_y", "center_y"))
        principal_point = _maybe_attr(model_parameters, ("principal_point",))
        if (cx is None or cy is None) and principal_point is not None:
            principal_arr = np.asarray(principal_point, dtype=np.float64).reshape(-1)
            if principal_arr.size >= 2:
                cx = float(principal_arr[0])
                cy = float(principal_arr[1])
        if fx is None or fy is None:
            angle_to_pixeldist_poly = _maybe_attr(model_parameters, ("angle_to_pixeldist_poly",))
            if angle_to_pixeldist_poly is not None:
                coeffs = np.asarray(angle_to_pixeldist_poly, dtype=np.float64).reshape(-1)
                if coeffs.size >= 2 and coeffs[1] > 0:
                    # FTheta cameras are approximated by the first-order angular scale.
                    fx = float(coeffs[1])
                    fy = float(coeffs[1])
        if fx is None or fy is None:
            pixeldist_to_angle_poly = _maybe_attr(model_parameters, ("pixeldist_to_angle_poly",))
            if pixeldist_to_angle_poly is not None:
                coeffs = np.asarray(pixeldist_to_angle_poly, dtype=np.float64).reshape(-1)
                if coeffs.size >= 2 and coeffs[1] > 0:
                    fx = float(1.0 / coeffs[1])
                    fy = float(1.0 / coeffs[1])
        if fx is None or fy is None or cx is None or cy is None:
            raise ValueError(
                f"Could not extract pinhole intrinsics for {camera_id}. "
                "Expected fx/fy/cx/cy or a 3x3 camera matrix on model_parameters."
            )
        fx = float(fx)
        fy = float(fy)
        cx = float(cx)
        cy = float(cy)

    return PinholeCamera(
        camera_id=camera_id,
        camera_model_id=camera_model_id,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def get_frame_timestamp_us(sensor: Any, frame_index: int) -> int:
    if hasattr(sensor, "get_frame_timestamp_us"):
        return int(sensor.get_frame_timestamp_us(frame_index))
    timestamps = np.asarray(getattr(sensor, "frames_timestamps_us"))
    if timestamps.ndim == 2:
        return int(timestamps[frame_index, -1])
    return int(timestamps[frame_index])


def get_frame_count(sensor: Any) -> int:
    if hasattr(sensor, "frames_count"):
        return int(sensor.frames_count)
    if hasattr(sensor, "get_frame_index_range"):
        frame_range = sensor.get_frame_index_range()
        return len(list(frame_range))
    timestamps = np.asarray(getattr(sensor, "frames_timestamps_us"))
    return int(timestamps.shape[0])


def selected_frame_indices(sensor: Any, frame_step: int, max_frames: int) -> list[int]:
    if frame_step <= 0:
        raise ValueError("--frame-step must be positive")
    if hasattr(sensor, "get_frame_index_range"):
        indices = list(sensor.get_frame_index_range())
    else:
        indices = list(range(get_frame_count(sensor)))
    selected = indices[::frame_step]
    if max_frames:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError("No frames selected. Check frame-step/max-frames and sensor data.")
    return [int(index) for index in selected]


def closest_frame_index(sensor: Any, timestamp_us: int) -> int:
    if hasattr(sensor, "get_closest_frame_index"):
        return int(sensor.get_closest_frame_index(int(timestamp_us)))
    count = get_frame_count(sensor)
    timestamps = np.asarray([get_frame_timestamp_us(sensor, index) for index in range(count)])
    return int(np.argmin(np.abs(timestamps.astype(np.int64) - int(timestamp_us))))


def get_sensor_c2w(sensor: Any, frame_index: int) -> np.ndarray:
    transform = sensor.get_frames_T_sensor_target("world", int(frame_index))
    arr = np.asarray(transform, dtype=np.float64)
    if arr.shape == (2, 4, 4):
        arr = arr[-1]
    return arr.reshape(4, 4)


def get_image_array(sensor: Any, frame_index: int) -> np.ndarray:
    if hasattr(sensor, "get_frame_image_array"):
        image = sensor.get_frame_image_array(int(frame_index))
    else:
        image = sensor.get_frame_image(int(frame_index))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8, copy=False)


def write_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        ok = cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"cv2.imwrite returned false for {path}")
    except ImportError:
        from PIL import Image

        Image.fromarray(image_rgb).save(path)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        with path.open("w", encoding="ascii") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {xyz.shape[0]}\n")
            for prop in ("x", "y", "z", "nx", "ny", "nz"):
                handle.write(f"property float {prop}\n")
            for prop in ("red", "green", "blue"):
                handle.write(f"property uchar {prop}\n")
            handle.write("end_header\n")
            for point, color in zip(xyz, rgb):
                handle.write(
                    f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                    f"0 0 0 {int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
        return

    normals = np.zeros_like(xyz, dtype=np.float32)
    vertex_data = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex_data[:] = list(map(tuple, np.concatenate([xyz, normals, rgb], axis=1)))
    PlyData([PlyElement.describe(vertex_data, "vertex")], text=False).write(path)


def write_cameras_txt(path: Path, cameras: Sequence[PinholeCamera]) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    for camera in cameras:
        lines.append(
            f"{camera.camera_model_id} PINHOLE {camera.width} {camera.height} "
            f"{camera.fx:.8f} {camera.fy:.8f} {camera.cx:.8f} {camera.cy:.8f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_images_txt(path: Path, images: Sequence[ColmapImage]) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(images)}",
    ]
    for image in images:
        c2w = np.asarray(image.c2w, dtype=np.float64).reshape(4, 4)
        r_wc = c2w[:3, :3]
        t_wc = c2w[:3, 3]
        r_cw = r_wc.T
        t_cw = -r_cw @ t_wc
        qvec = rotmat_to_qvec(r_cw)
        lines.append(
            f"{image.image_id} "
            f"{qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
            f"{t_cw[0]:.12f} {t_cw[1]:.12f} {t_cw[2]:.12f} "
            f"{image.camera_model_id} {image.image_name}"
        )
        lines.append("0.0 0.0 -1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rectify_camera_pair(
    *,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_camera: PinholeCamera,
    right_camera: PinholeCamera,
    left_c2w: np.ndarray,
    right_c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    image_size = (left_camera.width, left_camera.height)
    k_left = np.array(
        [[left_camera.fx, 0.0, left_camera.cx], [0.0, left_camera.fy, left_camera.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    k_right = np.array(
        [[right_camera.fx, 0.0, right_camera.cx], [0.0, right_camera.fy, right_camera.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    t_left_right = np.linalg.inv(right_c2w) @ left_c2w
    r_lr = t_left_right[:3, :3]
    # OpenCV >= 5 requires the translation as a column vector.
    t_lr = t_left_right[:3, 3].reshape(3, 1)
    zeros = np.zeros((5, 1), dtype=np.float64)
    r1, r2, p1, p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        k_left,
        zeros,
        k_right,
        zeros,
        image_size,
        r_lr,
        t_lr,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(k_left, zeros, r1, p1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(k_right, zeros, r2, p2, image_size, cv2.CV_32FC1)
    left_rect = cv2.remap(left_rgb, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_rgb, map2x, map2y, cv2.INTER_LINEAR)
    return left_rect, right_rect, p1, p2, r1


def unproject_rectified_to_world(
    *,
    disparity: np.ndarray,
    left_rect_rgb: np.ndarray,
    p_left: np.ndarray,
    p_right: np.ndarray,
    r_rect_left: np.ndarray,
    left_c2w: np.ndarray,
    min_disparity: float,
    max_depth_m: float,
    pixel_step: int,
    max_points_per_frame: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(disparity) & (disparity > min_disparity)
    if pixel_step > 1:
        yy = np.arange(disparity.shape[0])[:, None]
        xx = np.arange(disparity.shape[1])[None, :]
        valid &= (yy % pixel_step == 0) & (xx % pixel_step == 0)
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    fx = float(p_left[0, 0])
    fy = float(p_left[1, 1])
    cx = float(p_left[0, 2])
    cy = float(p_left[1, 2])
    baseline = abs(float(p_right[0, 3]) / max(float(p_right[0, 0]), 1e-6))
    depth = (fx * baseline) / disparity[ys, xs]
    depth_valid = np.isfinite(depth) & (depth > 0.0)
    if max_depth_m > 0:
        depth_valid &= depth <= max_depth_m
    ys = ys[depth_valid]
    xs = xs[depth_valid]
    depth = depth[depth_valid]
    if depth.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    if max_points_per_frame > 0 and depth.size > max_points_per_frame:
        keep = rng.choice(depth.size, size=max_points_per_frame, replace=False)
        ys = ys[keep]
        xs = xs[keep]
        depth = depth[keep]

    x_rect = (xs.astype(np.float64) - cx) * depth / fx
    y_rect = (ys.astype(np.float64) - cy) * depth / fy
    z_rect = depth.astype(np.float64)
    points_rect = np.stack([x_rect, y_rect, z_rect], axis=1)
    points_left = (np.asarray(r_rect_left, dtype=np.float64).T @ points_rect.T).T
    hom_left = np.concatenate([points_left, np.ones((points_left.shape[0], 1))], axis=1)
    points_world = (np.asarray(left_c2w, dtype=np.float64) @ hom_left.T).T[:, :3].astype(np.float32)
    colors = left_rect_rgb[ys, xs, :3].astype(np.uint8)
    return points_world, colors


def save_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_ncore_sequence_paths(ncore_root: Path, scene_id: str) -> list[Path]:
    root = Path(ncore_root)
    scene_dirs = [
        root / scene_id,
        root / f"pai_{scene_id}",
    ]

    for scene_dir in scene_dirs:
        if not scene_dir.is_dir():
            continue
        components = sorted(scene_dir.glob(f"pai_{scene_id}.ncore4*.zarr.itar"))
        if components:
            return components
        sequence_json = scene_dir / "sequence-ncore4.json"
        if sequence_json.exists():
            return [sequence_json]

    file_candidates = [
        root / f"pai_{scene_id}.ncore4.zarr.itar",
        root / f"{scene_id}.ncore4.zarr.itar",
    ]
    for candidate in file_candidates:
        if candidate.exists():
            return [candidate]

    glob_patterns = [
        f"**/*{scene_id}*.ncore4.zarr.itar",
        f"**/*{scene_id}*sequence*ncore4*.json",
        f"**/{scene_id}.json",
    ]
    for pattern in glob_patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches
    raise FileNotFoundError(
        f"Could not find an NCore sequence for scene_id={scene_id!r} under {root}"
    )


def resolve_ncore_sequence_path(ncore_root: Path, scene_id: str) -> Path:
    """Resolve one representative NCore path for backward-compatible callers."""

    return resolve_ncore_sequence_paths(ncore_root, scene_id)[0]


def load_ncore_sequence_loader(ncore_root: Path, scene_id: str) -> Any:
    try:
        from ncore.data.v4 import SequenceComponentGroupsReader, SequenceLoaderV4
    except ImportError as exc:
        raise RuntimeError(
            "NVIDIA NCore support requires `nvidia-ncore`. "
            "Install it in the Torch image or run inside vbogs-torch."
        ) from exc

    sequence_paths = resolve_ncore_sequence_paths(ncore_root, scene_id)
    group_reader = SequenceComponentGroupsReader(sequence_paths)
    return SequenceLoaderV4(group_reader)
