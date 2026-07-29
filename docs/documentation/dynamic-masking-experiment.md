# Dynamic-masking paired experiment

This experiment measures the causal effect of confirmed-moving-object masks on
Octree-AnyGS and VBOGS uncertainty. It compares an `unmasked` arm with a
`dynamic_masked` arm using the same NCore clip, cameras, split, settings, and
fixed upstream/VBGS seeds. Static-region held-out metrics are primary; full
frame metrics are retained for diagnosis.

Run from `vbogs-pipeline` after staging the Mask R-CNN weights:

```bash
scripts/experiment-dynamic-masking \
  --weights-path /workspace/VBOGS/data/models/maskrcnn_resnet50_fpn_v2.pth \
  --profile smoke
```

The smoke profile uses 48 primary-camera frames and 2,000 iterations. After
checking the overlays, static-mask coverage, filtered point clouds, and paired
report, run the fixed production profile:

```bash
scripts/experiment-dynamic-masking \
  --weights-path /workspace/VBOGS/data/models/maskrcnn_resnet50_fpn_v2.pth \
  --profile production
```

Artifacts are isolated by arm under
`data/experiments/dynamic-masking/<scene>/<profile>/<variant>/`, while curated
bundles and the final report are under
`outputs/experiments/dynamic-masking/<scene>/<profile>/`. The report contains
`comparison.json`, `metrics_table.md`, `per_view_deltas.csv`, and five fixed
primary-camera visual-review directories.

To resume after a completed training step, retain the arm workspace and rerun
with the desired `--stop-after`; `train_run.json` pins the downstream stages to
the exact Octree checkpoint. Do not mix artifacts between profiles or arms.

The mask artifact is built once per profile and evaluated against both arms.
It intentionally retains untracked, unmatchable, depth-unreliable, or otherwise
uncertain actors. Only confirmed movers are masked, so masking improvements must
be interpreted as a reduction in moving-object contamination rather than a
claim that every road user has been removed.
