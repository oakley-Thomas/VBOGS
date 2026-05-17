# Development Conventions

## Source Boundaries

Repo-owned code lives in:

```text
scripts/
vbogs/
tests/
docs/
configs/
docker/
```

The submodules are read-only dependencies:

```text
Octree-AnyGS/
vbgs/
```

Do not patch the submodules for VBOGS behavior. Wrap, subclass, or write
sibling functions in repo-owned code instead.

## Documentation Boundaries

This MkDocs site under `docs/` is the project documentation source of truth.
The top-level `README.md` is maintained by the human project owner and should
not be edited by agents.

When docs need to change:

- update a page under `docs/`;
- keep command examples runnable from the stated working directory;
- link to the exact detailed reference when a page gives only a summary;
- update `mkdocs.yml` navigation when adding a new page.

## Framework Boundary

PyTorch and JAX communicate through files only:

```text
Torch stage -> .npz/.npy/.json/.ply -> JAX stage
JAX stage   -> .npz/.npy/.json      -> Torch stage
```

Avoid cross-framework imports in shared helpers. Keep shared file-format logic
small and NumPy-based where possible.

## Coordinate Naming

Use explicit variable names:

| Name | Meaning |
| --- | --- |
| `points_world` | World-frame `xyz` plus RGB, used for bucketing |
| `points_norm` | Globally normalized `xyz + rgb`, used for VBGS |
| `anchor_pos` | Octree-AnyGS anchor positions |
| `anchor_level` | Octree-AnyGS anchor levels |

Frame mistakes silently produce plausible-looking but wrong outputs.

## Rendering Scalar Values

Octree-AnyGS obtains color from an MLP. There is no auxiliary per-anchor
channel to render directly. For scalar rendering, VBOGS reuses
`generate_gaussians` for geometry and substitutes scalar uncertainty before
calling rasterization. The implementation lives in `vbogs/render.py`.

## Adding a New Stage

New stages should:

1. live under `scripts/` for entry points and `vbogs/` for shared code;
2. declare their environment in docs and help text;
3. read and write explicit file contracts;
4. write metadata with provenance and key arguments;
5. have focused tests under `tests/`;
6. be added to pipeline docs if they become part of the main workflow.
