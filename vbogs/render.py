"""Scalar rendering helpers for VBOGS.

These functions live outside the Octree-AnyGS submodule and reuse its geometry
path without modifying upstream files.
"""

from __future__ import annotations

from typing import Dict

import torch


def _camera_intrinsics(viewpoint_camera) -> torch.Tensor:
    return torch.tensor(
        [
            [viewpoint_camera.fx, 0.0, viewpoint_camera.cx],
            [0.0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )


def render_scalar(
    viewpoint_camera,
    pc,
    per_anchor_scalar: torch.Tensor,
    *,
    iteration: int = 2_147_483_647,
    force_all_levels: bool = False,
) -> Dict[str, torch.Tensor]:
    """Render one scalar value per Octree-AnyGS anchor.

    The geometry/opacity/LOD selection is delegated to Octree-AnyGS. We replace
    the generated color tensor with a scalar channel broadcast from anchors to
    selected Gaussian offsets.
    """

    import gsplat

    if pc.gs_attr[-2:] != "3D":
        raise NotImplementedError("render_scalar currently supports 3D Gaussian attributes only")

    per_anchor_scalar = torch.as_tensor(per_anchor_scalar, dtype=torch.float32, device=pc.get_anchor.device)
    if per_anchor_scalar.ndim != 1:
        raise ValueError("per_anchor_scalar must be a 1D tensor")
    if per_anchor_scalar.shape[0] != pc.get_anchor.shape[0]:
        raise ValueError(
            f"Expected {pc.get_anchor.shape[0]} scalar values, got {per_anchor_scalar.shape[0]}"
        )

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

    visible_mask = pc._anchor_mask
    xyz, _color, opacity, scaling, rot, _sh_degree, selection_mask = pc.generate_gaussians(
        viewpoint_camera,
        visible_mask,
    )

    u_per_anchor = per_anchor_scalar[visible_mask]
    u_per_offset = u_per_anchor.repeat_interleave(pc.n_offsets)
    u_per_gaussian = u_per_offset[selection_mask].reshape(-1, 1)

    k_matrix = _camera_intrinsics(viewpoint_camera)
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)
    background = torch.zeros((1,), dtype=torch.float32, device=pc.get_anchor.device)

    render_colors, render_alphas, info = gsplat.rasterization(
        means=xyz,
        quats=rot,
        scales=scaling,
        opacities=opacity.squeeze(-1),
        colors=u_per_gaussian,
        viewmats=viewmat[None],
        Ks=k_matrix[None],
        backgrounds=background[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=None,
        render_mode=pc.render_mode,
    )

    unc_image = render_colors[0, ..., 0]
    alpha_image = render_alphas[0, ..., 0]
    radii = info["radii"].squeeze(0)
    return {
        "unc_image": unc_image,
        "alpha_image": alpha_image,
        "visible_mask": visible_mask,
        "selection_mask": selection_mask,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
