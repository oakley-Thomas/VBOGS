# Pipeline Overview

This page explains what each part of the VBOGS pipeline does, which runtime
owns it, and what files it passes to the next stage. It is a practical map of
the implemented pipeline. The authoritative algorithm specification remains the
repo source file `docs/manuscript/Algorithm.tex`.

## Big Picture

VBOGS starts with a trained Octree-AnyGS scene, adds a Bayesian uncertainty
head to each anchor, then renders those uncertainty values from candidate
camera poses to choose a next-best view.

```text
source data
    |
    v
prepare dataset
    |
    +--------------------------+
    |                          |
    v                          v
train Octree-AnyGS        export xyz+rgb points
    |                          |
    +------------+-------------+
                 |
                 v
        bucket points by anchor
                 |
                 v
        fit VBGS per anchor
                 |
                 v
       compute scalar U per anchor
                 |
                 v
  render uncertainty and score NBV poses
                 |
                 v
        bundle diagnostics and viewer export
```

The two major model families stay in separate containers:

| Runtime | Owns | Why |
| --- | --- | --- |
| `vbogs-torch` | Octree-AnyGS training, point export, bucketing, rendering, NBV scoring | Octree-AnyGS is PyTorch/CUDA based |
| `vbogs-jax` | VBGS fitting and uncertainty computation | VBGS is JAX/CUDA based |
| `vbogs-pipeline` | Orchestration, bundling, uploads | Runs the stage commands in the correct sibling service |
| `vbogs-preprocess` | Dataset preprocessing for formats that need it | Keeps video/COLMAP/360-camera tools out of the Torch and JAX images |

Data crosses runtime boundaries only through files on disk. There is no
in-process PyTorch-to-JAX tensor sharing.

## Operator Stages

`scripts/run_pipeline.sh` drives the implemented stages below. Use
`--start-at` and `--stop-after` to run a slice.

```text
prepare -> train -> stereo -> bucket -> fit -> inspect -> uncertainty
        -> map-viz -> render -> nbv -> nbv-viz -> bundle
```

| Stage | Service | Main entry point | Purpose |
| --- | --- | --- | --- |
| `prepare` | `vbogs-torch` or `vbogs-preprocess` | dataset-specific prepare scripts | Convert source data into the layout expected by the rest of the pipeline |
| `train` | `vbogs-torch` | `scripts/train_octree_anygs.py` | Train the Octree-AnyGS scene representation |
| `stereo` | `vbogs-torch` | `scripts/export_points_world.py` | Export colored world-frame points for the uncertainty head |
| `bucket` | `vbogs-torch` | `scripts/bucket_points.py` | Assign points to Octree-AnyGS anchors and normalize observations |
| `fit` | `vbogs-jax` | `scripts/fit_anchors.py` | Fit one VBGS posterior per observed anchor |
| `inspect` | `vbogs-jax` | posterior inspection helpers | Summarize fit quality before reducing to scalars |
| `uncertainty` | `vbogs-jax` | `scripts/compute_uncertainty.py` | Write one scalar uncertainty value per anchor |
| `map-viz` | `vbogs-torch` | map visualization helpers | Make static diagnostics for anchor uncertainty |
| `render` | `vbogs-torch` | `scripts/render_uncertainty_views.py` | Render RGB, uncertainty, and alpha images from saved views |
| `nbv` | `vbogs-torch` | `scripts/score_nbv.py` | Score candidate poses and choose the highest-scoring next-best view |
| `nbv-viz` | `vbogs-torch` | `scripts/visualize_m6.py` | Create NBV visual diagnostics |
| `bundle` | `vbogs-torch` | `scripts/bundle_run_outputs.py` | Collect diagnostics and local viewer files under `outputs/v1_0/<scene-id>/` |

## Stage 0: Prepare Dataset

The prepare stage adapts the selected dataset into a common scene contract.
The downstream training step expects posed RGB frames in an Octree-AnyGS
compatible COLMAP-style layout.

Examples:

| Dataset | Prepare behavior |
| --- | --- |
| KITTI-360 | Uses the native perspective stereo images, calibration, and ground-truth poses |
| NVIDIA PhysicalAI AV NCore | Decodes the selected clip and writes a scene-specific prepared layout |
| DJI Osmo 360 | Uses the preprocessing service for 360-camera perspective export and COLMAP-related tools |

Typical output:

```text
data/COLMAP/<scene-id>/
```

This stage is mostly data plumbing. It does not estimate uncertainty and does
not train the Octree-AnyGS scene.

## Stage 1: Train Octree-AnyGS

Training produces the primary scene representation: anchors, levels of detail,
opacity, geometry, and appearance parameters. Stereo or LiDAR depth is not used
as a training loss in the current VBOGS algorithm; depth observations are used
later by the uncertainty head.

```text
prepared posed RGB frames
    |
    v
Octree-AnyGS training
    |
    v
anchors + levels + learned splat parameters
```

Main output:

```text
/data/OCTREE-ANYGS/<scene-id>/<run>/
```

Later stages read real Octree-AnyGS fields such as `_anchor`, `_level`,
`voxel_size`, `fork`, `init_pos`, and `n_offsets`. There is no separate leaf
voxel object; an anchor at a level is the voxel for that level.

## Stage 2: Export World Points

The uncertainty head needs independent `xyz + rgb` observations. For stereo
datasets, VBOGS estimates disparity, converts disparity to depth, unprojects
pixels into the camera frame, transforms them into the world frame, and stores
their RGB color.

```text
left image + right image + calibration + pose
    |
    v
disparity -> depth -> camera xyz -> world xyz
    |
    v
points_world.npz: xyz, rgb, frame_id
```

The output point cloud is in world coordinates because bucketing must match
Octree-AnyGS's world-frame anchor grid.

Main output:

```text
data/points_world/<scene-id>/points_world.npz
```

Important fields:

| Field | Meaning |
| --- | --- |
| `xyz` | World-frame 3D point positions |
| `rgb` | Color sampled from the source image or projected camera |
| `frame_id` | Source frame for diagnostics and filtering |

## Stage 3a: Bucket Points Into Anchors

Bucketing connects the independent point observations to the Octree-AnyGS
scene. It uses world coordinates and the same grid discretization that
Octree-AnyGS used when it created anchors.

```text
Octree-AnyGS anchors              points_world.xyz
        |                               |
        v                               v
   level grid keys <------------- point grid keys
        |
        v
pts_by_anchor.npz
```

For each level `l`:

```text
cell_size(l) = voxel_size / fork^l
grid_coord(p, l) = round((p - init_pos) / cell_size(l))
```

A point is assigned to every Octree-AnyGS level that contains it, not only to
the finest matching level. This matters because Octree-AnyGS may render coarse
anchors from distant views.

The stage also writes globally normalized observations for VBGS:

```text
points_world = [xyz_world, rgb]
points_norm = normalize_data(points_world)
```

Main outputs:

```text
data/m4/<scene-id>/pts_by_anchor.npz
data/m4/<scene-id>/points_norm.npz
data/m4/<scene-id>/norm_params.json
data/m4/<scene-id>/bucket_metadata.json
```

The coordinate-frame split is deliberate:

| Array | Used for |
| --- | --- |
| `points_world` | Matching points to the Octree-AnyGS anchor grid |
| `points_norm` | VBGS fitting and comparable entropy values |

## Stage 3b: Fit VBGS Per Anchor

Each observed anchor receives a local VBGS mixture over its normalized 6D
observations. Anchors with too few points are marked unobserved and handled
later as high uncertainty.

```text
points_norm + pts_by_anchor
    |
    v
for each observed anchor:
    fit K_INIT components
    grow K while ELBO improves enough
    save posterior parameters
```

The default model-growth idea is:

```text
K_INIT -> 2*K_INIT -> ... -> K_MAX
accept larger K only when mean ELBO improves by ELBO_IMPROVEMENT_TOL
```

Main output:

```text
data/m4/<scene-id>/anchor_posterior.npz
```

This file stores posterior parameters such as mixture `alpha`, selected
component count, spatial Normal-Wishart parameters, color/delta parameters, and
an observed-anchor mask.

## Stage 4: Compute Scalar Uncertainty

The full VBGS posterior is too large and structured for the renderer. This
stage reduces each anchor's posterior to one scalar value `U[i]`.

```text
anchor_posterior.npz
    |
    v
posterior entropy per component
    |
    v
mixture-weighted scalar U per anchor
```

For observed anchors, VBOGS uses mixture-weighted per-component posterior
entropy:

```text
U_a = sum_k(expected_mixture_weight_ak * component_entropy_ak)
```

For unobserved anchors:

```text
U_a = U_MAX
```

Main outputs:

```text
data/m4/<scene-id>/U.npy
data/m4/<scene-id>/uncertainty_components.npz
data/m4/<scene-id>/uncertainty_metadata.json
```

`U.npy` is the key handoff from the JAX side back to the Torch rendering side.
Its length must match the number of Octree-AnyGS anchors.

## Stage 5: Render Uncertainty And Score NBV

The renderer reuses Octree-AnyGS geometry and level-of-detail selection, but
substitutes each anchor's uncertainty scalar for learned RGB color.

```text
Octree-AnyGS geometry + U.npy + candidate cameras
    |
    v
render_scalar(camera)
    |
    +--> uncertainty image
    +--> alpha image
    |
    v
score = sum(uncertainty_image) / (sum(alpha_image) + EPS)
```

The alpha-normalized score prefers poses that see uncertain content per unit
of visible surface. This differs from simply choosing the view with the most
visible scene area.

Main outputs:

```text
outputs/v1_0/<scene-id>/views/
outputs/v1_0/<scene-id>/nbv/
```

Important limitation: `render_scalar` only splats through existing
Octree-AnyGS anchors. Completely unseen empty space has no anchor and therefore
does not contribute to the NBV score.

## Stage 6: Bundle And View Results

The bundle stage collects the artifacts most useful for inspection and local
sharing. It also creates a portable local viewer export when not explicitly
disabled.

```text
model + prepared cameras + U.npy + diagnostics
    |
    v
outputs/v1_0/<scene-id>/
    |
    +--> <scene-id>.zip
    +--> local_viewer/
```

The local viewer export contains enough information to render RGB and
uncertainty without re-running the full pipeline:

| Bundle path | Purpose |
| --- | --- |
| `model/` | Octree-AnyGS checkpoint and patched config |
| `prepared/` | Camera metadata for train/test views |
| `uncertainty/U.npy` | Per-anchor uncertainty values |
| `VIEWER_COMMANDS.md` | Generated commands for that export |
| `local_viewer_manifest.json` | Source paths and export metadata |

See [Download and View Artifacts](local-artifact-viewing.md) for the full
local viewer workflow.

## Human Validation

M7 is a human validation pass, not just another script. Before treating a run
as trustworthy:

1. Inspect a sample of anchor posteriors after fitting.
2. Check that low-entropy anchors correspond to tight, well-observed geometry.
3. Check that high-entropy anchors correspond to sparse, noisy, distant, glass,
   or textureless regions.
4. Overlay `U` on held-out views.
5. Confirm the selected NBV pose matches scene intuition.
6. Record observed failure modes.

## Online Variant

The online ROS2 path reuses the same concepts with a fixed Octree-AnyGS scene.
It packages an offline state bundle, accepts incoming observations, updates
touched anchors, refreshes `U_online.npy`, and scores planner-provided
candidate poses.

```text
fixed Octree-AnyGS map + online VBGS state
    |
    v
incoming stereo/pose/candidates
    |
    v
bucket touched anchors -> JAX update -> refreshed U_online.npy
    |
    v
render candidate poses -> publish best pose
```

This online path updates uncertainty for existing anchors. It does not retrain
Octree-AnyGS online, densify the map, or solve the empty-space limitation.

## Debugging By Artifact

| If this looks wrong | Check first |
| --- | --- |
| Trained RGB render is poor | `train` outputs and Octree-AnyGS config |
| Point cloud floats away from the scene | dataset poses, calibration, and `points_world.npz` |
| Most anchors have no points | bucketing coordinate frame and anchor grid parameters |
| VBGS fits look noisy or unstable | point count histograms, `MIN_POINTS_PER_ANCHOR`, fit metadata |
| `U.npy` is mostly constant | observed mask, entropy metadata, `U_MAX` choice |
| Uncertainty render is blank | `U.npy` path, anchor count match, viewer/render service logs |
| NBV picks an odd pose | candidate pose set, alpha image, and score normalization |

## Related Pages

- [Quickstart](index.md)
- [Runtime Services](environments.md)
- [Render Server API](../running/realtime-viewer.md)
- [Download and View Artifacts](local-artifact-viewing.md)
