# Status and Milestones

The implementation status mirrors `PLAN.md` as of the current repo state.

## Completed Repo-Owned Milestones

| Milestone | Status | Notes |
| --- | --- | --- |
| M1 environment setup | Implemented | Docker services and conda helper commands exist |
| M2 train Octree-AnyGS | Implemented | Local dev and server profiles exist |
| M3 stereo to point cloud | Implemented | SGBM baseline; provider interface keeps room for RAFT or future matchers |
| M4a point to anchor bucketing | Implemented | Buckets points across all levels |
| M4b per-anchor VBGS fitting | Implemented | Batched JAX path plus loop fallback; full-scene quality pass still needed |
| M5 posterior to scalar uncertainty | Implemented | Writes `U.npy`, components, metadata, histogram |
| M6 scalar rendering and NBV scoring | Implemented | Diagnostic rendering and bundle stages available |

## Remaining Human Validation

M7 is not just a script. It requires domain judgment:

1. Pick a scene where expected uncertainty is known by inspection.
2. Run the full pipeline.
3. Inspect a sample of anchor posteriors after M4b.
4. Overlay `U` as heatmaps on held-out views.
5. Confirm the NBV pick matches intuition.
6. Document observed failure modes.

## Known Risks

| Risk | Current stance |
| --- | --- |
| Large anchor counts | Batched JAX fitting is implemented; use caps/smoke configs before full runs |
| Empty-region blindness | Known limitation; current NBV only sees existing anchors |
| ELBO across K is biased | Mean per-point ELBO is pragmatic; use held-out likelihood or BIC if selection looks wrong |
| Coordinate-frame mistakes | Bucketing must use world coordinates; fitting must use normalized coordinates |
| Score semantics | Current NBV score is alpha-normalized uncertainty per visible surface |

## Decisions Already Made

| Decision | Value |
| --- | --- |
| Stereo data source | KITTI-360 perspective stereo |
| Initial drive | `2013_05_28_drive_0008_sync` in the original plan; configs may target other drives |
| Baseline stereo matcher | OpenCV `StereoSGBM` |
| Training budget | 46 GB server cap; local dev profile is conservative |
| NBV candidate source | Planner-compatible interface with test/train/lattice first-pass sources |
| Initial uncertainty scalar | Mixture-weighted per-component posterior entropy |
