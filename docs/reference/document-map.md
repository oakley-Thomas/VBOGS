# Document Map

This page maps the main repo-owned documentation files. The normal entry point
for readers is the [site home](../index.md).

| Document | Use it for |
| --- | --- |
| [Quickstart](../getting-started/index.md) | Build containers, start the dev stack, and run the first pipeline commands |
| [Data Setup](../getting-started/data.md) | KITTI-360 paths, volumes, and download helpers |
| [Docker Pipeline](../running/docker-pipeline.md) | How `scripts/run_drive_pipeline.py` orchestrates stage slices |
| [Pipeline Arguments](../documentation/RUN_DRIVE_PIPELINE_ARGS.md) | Full `scripts/run_drive_pipeline.py` CLI/config reference |
| [Algorithm Overview](../algorithm/index.md) | Operator-facing explanation of the five-stage algorithm |
| [Algorithm.tex](../manuscript/Algorithm.tex) | Authoritative formatted algorithm specification and design constraints |
| [PER_ANCHOR_UNCERTAINTY_FORMULAS.tex](../manuscript/PER_ANCHOR_UNCERTAINTY_FORMULAS.tex) | Per-anchor uncertainty derivation and formulas |
| [Status and Milestones](../algorithm/status.md) | Implementation status and remaining validation work |
| [VBGS-Paper.pdf](../references/VBGS-Paper.pdf) | Local copy of the VBGS paper |
| [OctreeGS.pdf](../references/OctreeGS.pdf) | Local copy of the Octree-GS paper |

Files under `references/` are local paper copies and are not maintained as
repo-authored documentation.
