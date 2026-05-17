# Stage Details

This page maps the five algorithm stages to repo entry points, inputs, and
outputs.

## Stage 1: Train the Scene

Repo milestone: M2  
Entry point: `scripts/train_octree_anygs.py`  
Environment: `vbogs-torch`

Octree-AnyGS trains on posed RGB frames prepared in a COLMAP-style layout. The
result is the geometry, opacity, appearance, and anchor LOD state used by later
stages.

Main inputs:

- `/data/COLMAP/<drive>/`
- training profile from `configs/pipeline/*.yaml`

Main outputs:

- `/data/OCTREE-ANYGS/<drive>/<timestamp>/config.yaml`
- `/data/OCTREE-ANYGS/<drive>/<timestamp>/point_cloud/iteration_*/`

## Stage 2: Stereo Point Cloud

Repo milestone: M3  
Entry point: `scripts/stereo_to_pointcloud.py`  
Environment: `vbogs-torch`

For each rectified stereo pair, VBOGS estimates disparity, converts disparity
to depth, unprojects points into the left camera frame, transforms them to
world coordinates, attaches RGB, and concatenates the result.

Main inputs:

- `data/KITTI-360/images/<drive>/image_00/data_rect/`
- `data/KITTI-360/images/<drive>/image_01/data_rect/`
- `data/KITTI-360/data_poses/<drive>/cam0_to_world.txt`
- `data/KITTI-360/calibration/perspective.txt`

Main outputs:

- `data/points_world/<drive>/points_world.npz`
- optional `data/points_world/<drive>/points_world.ply`
- metadata sidecars

The NPZ contains `xyz`, `rgb`, and `frame_id`.

## Stage 3: Per-Anchor VBGS Fitting

Repo milestones: M4a and M4b  
Entry points: `scripts/bucket_points.py`, `scripts/fit_anchors.py`  
Environments: `vbogs-torch`, then `vbogs-jax`

M4a buckets world-frame points into every Octree-AnyGS level whose anchor cell
contains the point. It also writes normalized points for VBGS.

M4b fits a VBGS mixture for every observed anchor. The component count starts
at `K_INIT` and grows while mean per-point ELBO improves by at least
`ELBO_IMPROVEMENT_TOL`.

Main M4a outputs:

- `data/m4/<drive>/points_norm.npz`
- `data/m4/<drive>/pts_by_anchor.npz`
- `data/m4/<drive>/norm_params.json`
- `data/m4/<drive>/bucket_metadata.json`

Main M4b outputs:

- `data/m4/<drive>/anchor_posterior.npz`
- fit metadata and shard metadata when sharded fitting is used

## Stage 4: Scalar Uncertainty

Repo milestone: M5  
Entry point: `scripts/compute_uncertainty.py`  
Environment: `vbogs-jax` or pure NumPy-compatible Python

VBOGS computes mixture-weighted per-component posterior entropy for observed
anchors. Unobserved anchors receive `U_MAX`, which defaults to the maximum
finite observed uncertainty unless overridden.

Main outputs:

- `data/m4/<drive>/U.npy`
- `data/m4/<drive>/uncertainty_components.npz`
- `data/m4/<drive>/uncertainty_metadata.json`
- optional `data/m4/<drive>/uncertainty_histogram.png`

## Stage 5: Next-Best View Selection

Repo milestone: M6  
Entry points: `scripts/render_uncertainty_views.py`, `scripts/score_nbv.py`,
`scripts/visualize_m6.py`  
Environment: `vbogs-torch`

The renderer reuses Octree-AnyGS geometry and LOD selection, substitutes the
per-anchor scalar `U` for color, then rasterizes uncertainty and alpha images.
Candidate poses are scored as:

```text
sum(uncertainty_image) / (sum(alpha_image) + EPS)
```

Main outputs:

- rendered RGB/uncertainty diagnostics under `outputs/v1_0/<drive>/views/`
- NBV scores and top images under `outputs/v1_0/<drive>/nbv/`
- curated bundle under `outputs/v1_0/<drive>/`

## Stage Dependencies

```text
M1 -> M2 -> M4a -> M4b -> M5 -> M6 -> M7
      M3 ----^
```

M2 and M3 can run independently after environment setup. M4a onward is linear.
