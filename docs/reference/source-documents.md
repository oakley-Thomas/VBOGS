# Source Documents

These source documents explain the design and implementation contract behind
the documentation site.

## Authoritative Algorithm Sources

| Document | Use |
| --- | --- |
| [Algorithm.tex](../manuscript/Algorithm.tex) | Authoritative five-stage algorithm specification |
| [PER_ANCHOR_UNCERTAINTY_FORMULAS.tex](../manuscript/PER_ANCHOR_UNCERTAINTY_FORMULAS.tex) | Detailed per-anchor uncertainty derivation |
| [Algorithm overview](../documentation/ALGORITHM_OVERVIEW.md) | Plain-language narrative walkthrough |

## Operator References

| Document | Use |
| --- | --- |
| [Pipeline arguments](../documentation/RUN_DRIVE_PIPELINE_ARGS.md) | Exhaustive `scripts/run_drive_pipeline.py` CLI/config reference |
| [Command reference](commands.md) | Common command snippets and direct stage entry points |
| [Artifacts and data layout](artifacts.md) | File contracts by stage |

## External Paper Copies

The repository includes local copies of:

- [VBGS-Paper.pdf](../references/VBGS-Paper.pdf)
- [OctreeGS.pdf](../references/OctreeGS.pdf)

These are reference papers, not repo-authored docs.

## Project Rules for Agents

Keep implementation and documentation consistent with these conventions:

- Do not edit `Octree-AnyGS/` or `vbgs/`; they are submodules.
- Do not edit the top-level `README.md`; it is maintained by the project
  owner.
- Put repo-authored documentation under `docs/`.
- If an implementation request conflicts with `Algorithm.tex`, flag it before
  diverging.
