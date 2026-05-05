"""Rendering helpers for VBOGS uncertainty diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class ScalarRenderResult:
    """Container that works for both tuple-unpack and mapping-style callers."""

    unc_image: "torch.Tensor"
    alpha_image: "torch.Tensor"
    visible_mask: "torch.Tensor"
    selection_mask: "torch.Tensor"
    visibility_filter: "torch.Tensor"
    radii: "torch.Tensor"

    def __iter__(self) -> Iterator["torch.Tensor"]:
        yield self.unc_image
        yield self.alpha_image

    def __getitem__(self, key: str | int) -> "torch.Tensor":
        if key == 0:
            return self.unc_image
        if key == 1:
            return self.alpha_image
        if isinstance(key, str):
            return getattr(self, key)
        raise KeyError(key)


def expand_anchor_scalars_to_gaussians(
    per_anchor_scalar: "torch.Tensor",
    visible_mask: "torch.Tensor",
    selection_mask: "torch.Tensor",
    n_offsets: int,
) -> "torch.Tensor":
    """Broadcast per-anchor scalars to the selected Gaussian offsets."""

    import torch

    if per_anchor_scalar.ndim != 1:
        raise ValueError("per_anchor_scalar must be a 1D tensor")
    if visible_mask.ndim != 1 or visible_mask.dtype != torch.bool:
        raise ValueError("visible_mask must be a 1D bool tensor")
    if selection_mask.ndim != 1 or selection_mask.dtype != torch.bool:
        raise ValueError("selection_mask must be a 1D bool tensor")
    if per_anchor_scalar.shape[0] != visible_mask.shape[0]:
        raise ValueError(
            "per_anchor_scalar length must match the full anchor count "
            f"({per_anchor_scalar.shape[0]} != {visible_mask.shape[0]})"
        )
    if n_offsets <= 0:
        raise ValueError("n_offsets must be positive")

    visible_scalars = per_anchor_scalar[visible_mask]
    expanded = visible_scalars.repeat_interleave(int(n_offsets))
    if expanded.shape[0] != selection_mask.shape[0]:
        raise ValueError(
            "selection_mask length must match visible anchors expanded by n_offsets "
            f"({selection_mask.shape[0]} != {expanded.shape[0]})"
        )
    return expanded[selection_mask][:, None]


def _camera_intrinsics(viewpoint_camera: Any, device: "torch.device") -> "torch.Tensor":
    import torch

    return torch.tensor(
        [
            [viewpoint_camera.fx, 0.0, viewpoint_camera.cx],
            [0.0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def _zero_background(device: "torch.device") -> "torch.Tensor":
    import torch

    return torch.zeros(1, dtype=torch.float32, device=device)


def render_scalar(
    viewpoint_camera: Any,
    pc: Any,
    pipe_or_scalar: Any,
    per_anchor_scalar: "torch.Tensor | None" = None,
    iteration: int = 2_147_483_647,
    *,
    force_all_levels: bool = False,
) -> ScalarRenderResult:
    """Render a per-anchor scalar through Octree-AnyGS geometry.

    Supports both call shapes used in the repo:

    - `render_scalar(cam, pc, per_anchor_scalar, iteration=...)`
    - `render_scalar(cam, pc, pipe, per_anchor_scalar, iteration)`
    """

    import gsplat
    import torch

    if per_anchor_scalar is None:
        pipe = None
        per_anchor_scalar = pipe_or_scalar
    else:
        pipe = pipe_or_scalar

    if per_anchor_scalar.ndim != 1:
        raise ValueError("per_anchor_scalar must be 1D")

    anchor_count = int(pc.get_anchor.shape[0])
    if int(per_anchor_scalar.shape[0]) != anchor_count:
        raise ValueError(
            f"per_anchor_scalar has {per_anchor_scalar.shape[0]} values, "
            f"but the scene has {anchor_count} anchors"
        )

    device = pc.get_anchor.device
    per_anchor_scalar = per_anchor_scalar.to(device=device, dtype=torch.float32)

    if force_all_levels and hasattr(pc, "set_anchor_mask_perlevel"):
        pc.set_anchor_mask_perlevel(
            viewpoint_camera.camera_center,
            viewpoint_camera.resolution_scale,
            pc.levels - 1,
        )
    else:
        pc.set_anchor_mask(
            viewpoint_camera.camera_center,
            iteration,
            viewpoint_camera.resolution_scale,
        )

    if pipe is not None and getattr(pipe, "add_prefilter", False):
        from gaussian_renderer.render import prefilter_voxel

        visible_mask = prefilter_voxel(viewpoint_camera, pc).squeeze()
    else:
        visible_mask = pc._anchor_mask

    xyz, _color, opacity, scaling, rot, _sh_degree, selection_mask = pc.generate_gaussians(
        viewpoint_camera,
        visible_mask,
    )
    scalar_color = expand_anchor_scalars_to_gaussians(
        per_anchor_scalar,
        visible_mask,
        selection_mask,
        int(pc.n_offsets),
    )

    k_matrix = _camera_intrinsics(viewpoint_camera, device)
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)
    background = _zero_background(device)
    gs_attr = getattr(pc, "gs_attr", "3D")

    if gs_attr[-2:] == "3D":
        render_colors, render_alphas, info = gsplat.rasterization(
            means=xyz,
            quats=rot,
            scales=scaling,
            opacities=opacity.squeeze(-1),
            colors=scalar_color,
            viewmats=viewmat[None],
            Ks=k_matrix[None],
            backgrounds=background[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            sh_degree=None,
            render_mode=pc.render_mode,
        )
    elif gs_attr[-2:] == "2D":
        raster_result, info = gsplat.rasterization_2dgs(
            means=xyz,
            quats=rot,
            scales=scaling,
            opacities=opacity.squeeze(-1),
            colors=scalar_color,
            viewmats=viewmat[None],
            Ks=k_matrix[None],
            backgrounds=background[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            sh_degree=None,
            render_mode=pc.render_mode,
        )
        render_colors, render_alphas = raster_result[:2]
    else:
        raise ValueError(f"Unknown gs_attr: {gs_attr}")

    radii = info.get("radii")
    if radii is None:
        radii = torch.ones((xyz.shape[0],), dtype=torch.bool, device=device)
    else:
        radii = radii.squeeze(0)

    return ScalarRenderResult(
        unc_image=render_colors[0, ..., 0],
        alpha_image=render_alphas[0, ..., 0],
        visible_mask=visible_mask,
        selection_mask=selection_mask,
        visibility_filter=radii > 0,
        radii=radii,
    )
