"""Camera adapters used by the realtime Octree-AnyGS viewer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def camera_to_c2w(cam: Any) -> np.ndarray:
    """Return a row-major camera-to-world matrix from an Octree camera object."""

    matrix = cam.world_view_transform.transpose(0, 1).inverse()
    return matrix.detach().cpu().numpy().astype(np.float32)


def coerce_c2w(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected c2w matrix shape (4, 4), got {matrix.shape}")
    return matrix


def _fallback_projection_matrix(znear: float, zfar: float, fovx: float, fovy: float) -> np.ndarray:
    tan_half_x = math.tan(float(fovx) * 0.5)
    tan_half_y = math.tan(float(fovy) * 0.5)
    top = tan_half_y * znear
    bottom = -top
    right = tan_half_x * znear
    left = -right

    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = 2.0 * znear / (right - left)
    matrix[1, 1] = 2.0 * znear / (top - bottom)
    matrix[0, 2] = (right + left) / (right - left)
    matrix[1, 2] = (top + bottom) / (top - bottom)
    matrix[3, 2] = 1.0
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    return matrix


def _projection_tensor(znear: float, zfar: float, fovx: float, fovy: float, device: str):
    import torch

    try:
        from utils.graphics_utils import getProjectionMatrix

        return getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).to(device)
    except Exception:
        matrix = _fallback_projection_matrix(znear, zfar, fovx, fovy)
        return torch.tensor(matrix, dtype=torch.float32, device=device).transpose(0, 1)


@dataclass
class ViewerCamera:
    """Small camera object compatible with Octree-AnyGS render functions."""

    source_cam: Any
    c2w_np: np.ndarray
    uid: int
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    znear: float
    zfar: float
    device: str = "cuda"

    def __post_init__(self) -> None:
        import torch

        self.c2w_np = coerce_c2w(self.c2w_np)
        self.uid = int(self.uid)
        self.image_name = str(self.image_name)
        self.image_path = getattr(self.source_cam, "image_path", "")
        self.resolution_scale = float(getattr(self.source_cam, "resolution_scale", 1.0))
        self.image_width = int(self.width)
        self.image_height = int(self.height)
        self.fx = float(self.fx)
        self.fy = float(self.fy)
        self.cx = float(self.cx)
        self.cy = float(self.cy)
        self.znear = float(self.znear)
        self.zfar = float(self.zfar)
        self.FoVx = focal_to_fov(self.fx, self.image_width)
        self.FoVy = focal_to_fov(self.fy, self.image_height)

        world_to_view = np.linalg.inv(self.c2w_np).astype(np.float32)
        self.world_view_transform = torch.tensor(world_to_view, dtype=torch.float32, device=self.device).transpose(0, 1)
        self.projection_matrix = _projection_tensor(self.znear, self.zfar, self.FoVx, self.FoVy, self.device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()

    @classmethod
    def from_source(
        cls,
        source_cam: Any,
        *,
        c2w: Any | None = None,
        uid: int | None = None,
        image_name: str | None = None,
        device: str = "cuda",
    ) -> "ViewerCamera":
        source_c2w = camera_to_c2w(source_cam) if c2w is None else coerce_c2w(c2w)
        width = int(getattr(source_cam, "image_width"))
        height = int(getattr(source_cam, "image_height"))
        fx = float(getattr(source_cam, "fx"))
        fy = float(getattr(source_cam, "fy"))
        cx = float(getattr(source_cam, "cx", width * 0.5))
        cy = float(getattr(source_cam, "cy", height * 0.5))
        return cls(
            source_cam=source_cam,
            c2w_np=source_c2w,
            uid=int(getattr(source_cam, "uid", 0) if uid is None else uid),
            image_name=str(getattr(source_cam, "image_name", "viewer") if image_name is None else image_name),
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            znear=float(getattr(source_cam, "znear", 0.01)),
            zfar=float(getattr(source_cam, "zfar", 100.0)),
            device=device,
        )
