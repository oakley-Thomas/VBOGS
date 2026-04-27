# How the VBOGS Algorithm Works

VBOGS builds a Gaussian-splatting scene that knows not only what the world looks
like, but also where its model is uncertain. It combines Octree-AnyGS for the
main scalable scene representation with VBGS for per-anchor Bayesian
uncertainty, then uses that uncertainty to choose the next camera view that
should be most informative.

The algorithm has five stages.

## 1. Train the Octree-AnyGS Scene

First, Octree-AnyGS is trained normally on posed RGB frames. This produces a
level-of-detail scene made of anchors, where each anchor is effectively a voxel
cell at some octree level. The project does not change upstream Octree-AnyGS
training here: the result is the geometry, opacity, and appearance model used
later for rendering and visibility.

## 2. Build a Stereo Point Cloud

For each stereo image pair, the algorithm estimates disparity, converts it to
depth, unprojects depth pixels into the left camera frame, and transforms those
points into world coordinates. Each point keeps its RGB value, producing a
six-dimensional point cloud: `xyz + rgb`.

The point cloud is kept in two coordinate systems for two different jobs:

- `points_world` is used for bucketing points into Octree-AnyGS anchors, because
  anchor cells are defined in world space.
- `points_norm` is used for VBGS fitting, because VBGS expects normalized data.

Keeping this distinction explicit is essential. Bucketing in normalized
coordinates would no longer match the Octree-AnyGS grid.

## 3. Fit a VBGS Posterior Per Anchor

Each stereo point is assigned to every Octree-AnyGS anchor cell that contains it,
across all levels of detail. This matters because Octree-AnyGS may render a
coarse anchor from a distant camera and a finer anchor from a nearby camera. If
only the finest anchor received points, coarse anchors would look falsely
uncertain even in well-observed areas.

For every anchor with enough assigned points, VBOGS fits a Variational Bayes
Gaussian mixture model using the normalized `xyz + rgb` samples. The number of
mixture components starts at `K_INIT` and grows while the per-point ELBO improves
enough to justify the larger model. Anchors with too few points are marked as
unobserved.

The output of this stage is a posterior distribution per anchor, not a rendered
image. The posterior captures how confidently the model explains the local
geometry and color samples inside that anchor cell.

## 4. Convert Each Posterior to One Uncertainty Value

The VBGS posterior is reduced to a scalar uncertainty `U[i]` for each anchor.
Observed anchors use a mixture-weighted entropy of the posterior components:
components with high uncertainty and high expected mixture weight contribute
more to the final value. Unobserved anchors receive `U_MAX`.

These entropy values are computed in normalized coordinates, so anchor
uncertainties can be compared consistently across the scene.

## 5. Score Candidate Next-Best Views

Finally, VBOGS evaluates candidate camera poses. For each candidate, it renders
the per-anchor uncertainty values through Octree-AnyGS's existing
level-of-detail Gaussian geometry. Instead of rendering learned color, the
renderer substitutes each anchor's scalar uncertainty as the Gaussian color.

This produces:

- an uncertainty image, showing how much uncertain surface the camera would see;
- an alpha image, showing how much actual scene surface is visible.

Each candidate pose receives an alpha-normalized score:

```text
score = sum(uncertainty_image) / (sum(alpha_image) + EPS)
```

The best next view is the candidate with the highest score. Normalizing by alpha
means the selected pose is biased toward seeing uncertain content, not merely
toward seeing a large amount of already-confident geometry.

## Important Limitation

The current algorithm can only score uncertainty on anchors that already exist
in the Octree-AnyGS scene. Truly unseen empty space contributes nothing, because
there is no Gaussian to render there. As a result, VBOGS chooses views that
improve poorly modeled known surfaces; it does not yet perform full exploration
of never-observed volume.
