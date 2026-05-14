# How the Colored Point Cloud Becomes Uncertainty

This note explains how VBOGS uses the colored stereo point cloud to calculate
per-anchor uncertainty. It follows the implemented pipeline from point-cloud
generation through VBGS fitting and final `U.npy` export. For the full
pseudocode, see [Algorithm.txt](Algorithm.txt).

## Summary

The colored point cloud starts as world-frame 6D samples:

```text
[x_world, y_world, z_world, r, g, b]
```

Those samples are globally normalized for VBGS fitting, while the original
world-frame `xyz` is kept for anchor bucketing. Each sample is assigned to the
Octree-AnyGS anchors whose grid cells contain its world-space position. For each
observed anchor, VBOGS fits a VBGS mixture model over the anchor's local
normalized 6D samples. The fitted Bayesian posterior is then reduced to one
scalar uncertainty value per anchor.

The color channels are not only saved for visualization. They participate in
the VBGS likelihood, so the posterior reflects both geometric uncertainty and
color/material consistency inside each anchor.

## 1. Build the Colored Point Cloud

`scripts/stereo_to_pointcloud.py` reads KITTI-360 stereo image pairs and camera
poses. For each valid stereo pixel, it:

1. estimates disparity;
2. converts disparity to depth;
3. unprojects the pixel into the camera frame;
4. transforms the point into world coordinates;
5. copies the RGB value from the left image.

The output artifact is `points_world.npz`, with:

| Key | Shape | Meaning |
| --- | --- | --- |
| `xyz` | `(N, 3)` | World-frame point positions |
| `rgb` | `(N, 3)` | RGB colors from source pixels |
| `frame_id` | `(N,)` | Source frame for each point |

## 2. Bucket Points Into Octree-AnyGS Anchors

`scripts/bucket_points.py` combines position and color into one 6D array:

```python
points_world = concatenate([xyz_world, rgb], axis=1)
points_norm = normalize_data(points_world)
```

Two coordinate versions are kept because they serve different purposes:

| Data | Used for | Reason |
| --- | --- | --- |
| `xyz_world` | Anchor bucketing | Octree-AnyGS anchors are defined in world space |
| `points_norm` | VBGS fitting | VBGS expects globally normalized feature vectors |

Bucketing is done level by level. For each Octree-AnyGS level, the script
recomputes grid coordinates for both anchors and points:

```text
grid_coord = round((xyz_world - init_pos) / cell_size)
cell_size = voxel_size / fork^level
```

If a point's grid coordinate matches an anchor grid coordinate, that point is
assigned to that anchor. This is done across all levels, not only the finest
level, because Octree-AnyGS may render coarse anchors for distant views.

The main outputs are:

| Artifact | Meaning |
| --- | --- |
| `points_norm.npz` | Normalized 6D samples plus original world/color fields |
| `pts_by_anchor.npz` | Packed point indices for each anchor |
| `norm_params.json` | Global normalization offset and scale |

## 3. Fit a VBGS Posterior Per Anchor

`scripts/fit_anchors.py` loads `points_norm.npz` and `pts_by_anchor.npz`. For
each anchor with at least `MIN_POINTS_PER_ANCHOR`, it gathers that anchor's
normalized 6D rows and fits a VBGS mixture model.

VBGS treats the 6D sample as two coupled pieces:

```text
spatial part = points_norm[:, :3]
color part   = points_norm[:, -3:]
```

During fitting, component assignment uses:

```text
space_logprob + color_logprob + prior_logprob
```

This means points are grouped by both where they are and what color they are.
If an anchor contains geometrically or visually inconsistent samples, the
posterior can become broader or require more mixture components.

The saved posterior includes:

| Field | Meaning |
| --- | --- |
| `alpha` | Dirichlet mixture counts/weights |
| `spatial_mean`, `spatial_kappa`, `spatial_u`, `spatial_n` | Spatial Normal-Wishart posterior |
| `delta_mean`, `delta_kappa`, `delta_u`, `delta_n` | Color/delta posterior |
| `point_count`, `observed_anchor_ids` | Assignment counts and observed-anchor lookup |
| `final_k` | Selected component count for the anchor |
| `is_observed`, `fit_completed` | Whether the anchor had enough points and completed fitting |

## 4. Convert the Posterior to Scalar Uncertainty

`scripts/compute_uncertainty.py` reduces each observed anchor posterior to one
float value.

For each active mixture component, it computes:

```text
component_entropy_k = spatial_entropy_k + color_delta_entropy_k
```

Then it normalizes the component weights from `alpha`:

```text
weight_k = alpha_k / sum_j alpha_j
```

The final anchor uncertainty is:

```text
U_anchor = sum_k weight_k * component_entropy_k
```

Unobserved anchors receive `U_MAX`. By default, `U_MAX` is the maximum finite
observed uncertainty in the scene, unless the script is run with an explicit
`--u-max`.

The main output is:

| Artifact | Meaning |
| --- | --- |
| `U.npy` | One uncertainty scalar per Octree-AnyGS anchor |
| `uncertainty_components.npz` | Diagnostic entropy components and weights |
| `uncertainty_metadata.json` | Summary statistics and provenance |

## Important Detail: Dirichlet Entropy

The script also computes Dirichlet entropy from `alpha` as a diagnostic, but it
does not add it to `U_anchor`. The current project choice is:

```text
pi-weighted per-component entropy
```

not total mixture entropy.

## 5. Use Uncertainty for Next-Best View Scoring

After `U.npy` is created, `vbogs/render.py` renders per-anchor uncertainty
through Octree-AnyGS geometry. It reuses Octree-AnyGS Gaussian generation, but
substitutes each anchor's scalar uncertainty where normal RGB color would be.

`scripts/score_nbv.py` scores candidate camera poses with:

```text
score = sum(uncertainty_image) / (sum(alpha_image) + EPS)
```

The winning next-best view is the candidate that sees the highest
alpha-normalized uncertainty.

## What Color Contributes

Color affects uncertainty in two places:

1. During VBGS fitting, RGB contributes to the likelihood used for component
   assignment.
2. During uncertainty computation, the color/delta posterior contributes its
   entropy to each component's uncertainty.

So an anchor can be uncertain because its geometry is poorly constrained, its
color/material observations are inconsistent, or both.

## Known Limitation

The uncertainty is attached only to existing Octree-AnyGS anchors. Completely
unobserved empty space has no Gaussian to render, so it does not directly
attract next-best views in the current implementation.
