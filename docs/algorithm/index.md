# Algorithm Overview

VBOGS is a five-stage pipeline:

1. Train an Octree-AnyGS scene from posed RGB frames.
2. Build a world-frame stereo point cloud with RGB values.
3. Assign points to Octree-AnyGS anchors and fit a VBGS posterior per anchor.
4. Reduce each posterior to one scalar uncertainty value.
5. Render those anchor uncertainties through the scene and score next-best
   views.

The authoritative formatted specification is
[Algorithm.tex](../manuscript/Algorithm.tex). This page is the operator-facing
summary.

## Components

| Component | Role |
| --- | --- |
| Octree-AnyGS | Primary scalable Gaussian-splatting scene representation with levels of detail |
| Stereo point cloud | Observed `xyz + rgb` samples used to fit uncertainty |
| VBGS | Per-anchor Bayesian Gaussian mixture posterior |
| `render_scalar` | Geometry reuse path that renders uncertainty instead of learned color |
| NBV scorer | Selects the candidate pose with maximum alpha-normalized expected uncertainty |

## Key Terms

Anchor
: Octree-AnyGS stores anchors as a flat tensor with a level. In this project,
  "anchor" and "voxel" mean the same anchor cell at that level.

Per-anchor uncertainty
: A scalar `U[i]` derived from the VBGS posterior for points assigned to anchor
  `i`.

Next-best view
: The candidate camera pose with the highest alpha-normalized rendered
  uncertainty score.

Observed anchor
: An anchor with at least `MIN_POINTS_PER_ANCHOR` assigned points and a
  completed VBGS fit.

Unobserved anchor
: An anchor below the point threshold or without a completed fit. It receives
  `U_MAX`.

## Coordinate Frames

VBOGS keeps two point arrays because they have different jobs:

| Array | Coordinate frame | Used for |
| --- | --- | --- |
| `points_world` | World coordinates plus RGB | Matching points to Octree-AnyGS anchor grid cells |
| `points_norm` | Globally normalized `xyz + rgb` | VBGS fitting and entropy comparisons |

Bucketing in normalized coordinates will not match Octree-AnyGS anchors.
Fitting in raw world coordinates makes VBGS priors and entropy scales harder to
compare.

## Known Limitation

`render_scalar` splats uncertainty through existing Octree-AnyGS anchors only.
Truly unseen empty space has no anchor to render, so it contributes zero to the
NBV score. The current system chooses views that improve poorly modeled known
surfaces, not full exploration of never-seen volume.
