# Uncertainty Formula Summary

The detailed derivation lives in
[PER_ANCHOR_UNCERTAINTY_FORMULAS.tex](../manuscript/PER_ANCHOR_UNCERTAINTY_FORMULAS.tex).
This page summarizes what the implementation computes.

## Observations

Each stereo point is a six-dimensional observation:

```text
x_world = [xyz_world, rgb]
```

The normalized version is:

```text
y = (x_world - global_offset) / global_stdev
```

Bucketing uses `xyz_world`; VBGS fitting uses normalized `y`.

## Point-to-Anchor Assignment

For an anchor level `l`:

```text
cell_size(l) = voxel_size / fork^l
grid_coord(p, l) = round((p - init_pos) / cell_size(l))
```

A point belongs to an anchor when their grid coordinates match at that level.
Points are assigned across all levels, not only the finest level.

## Component Growth

For each observed anchor, the mixture starts at `K_INIT`. Larger mixtures are
accepted when:

```text
mean_elbo(K_next) - mean_elbo(K) >= ELBO_IMPROVEMENT_TOL
```

The default hyperparameters from the implementation plan are:

| Hyperparameter | Value |
| --- | --- |
| `K_INIT` | `pc.n_offsets`, currently `10` |
| `K_MAX` | `4 * K_INIT`, currently `40` |
| `K_GROWTH_FACTOR` | `2` |
| `MIN_POINTS_PER_ANCHOR` | `20` |
| `ELBO_IMPROVEMENT_TOL` | `0.01` nats/point |

## Mixture Weights

For active component `k` in anchor `a`, the expected mixture weight is:

```text
w_ak = alpha_ak / sum_j(alpha_aj)
```

These are VBGS mixture weights. They are not Octree-AnyGS opacity values.

## Per-Component Entropy

Each component contributes:

```text
H_ak = H_spatial_ak + H_color_ak
```

`H_spatial_ak` is the Normal-Wishart posterior entropy for the spatial
Gaussian parameters. `H_color_ak` is the delta/color multivariate-normal
entropy.

These entropies are computed in globally normalized coordinates so they remain
comparable across anchors.

## Final Anchor Uncertainty

The observed-anchor uncertainty is:

```text
U_a = sum_k(w_ak * H_ak)
```

Unobserved anchors receive:

```text
U_a = U_MAX
```

By default:

```text
U_MAX = max finite observed U
```

Override it with `scripts/compute_uncertainty.py --u-max` or
`scripts/run_drive_pipeline.py --uncertainty-u-max`.

## Diagnostic Dirichlet Entropy

The implementation also computes Dirichlet entropy for diagnostics and saves it
in `uncertainty_components.npz`. It is not added to the final `U.npy` scalar.

## Implementation Mapping

| Formula quantity | Implementation field/function |
| --- | --- |
| `alpha_ak` | `alpha` |
| final component count | `final_k` |
| `w_ak` | `alpha / sum(alpha)` over active components |
| spatial kappa | `spatial_kappa` |
| spatial Wishart scale | `spatial_u` |
| spatial degrees of freedom | `spatial_n` |
| color/delta kappa | `delta_kappa` |
| color/delta scale | `delta_u` |
| color/delta degrees of freedom | `delta_n` |
| spatial entropy | `normal_wishart_entropy(...)` |
| color entropy | `delta_mvn_entropy(...)` |
| final uncertainty | `uncertainty[anchor_id]` in `U.npy` |
