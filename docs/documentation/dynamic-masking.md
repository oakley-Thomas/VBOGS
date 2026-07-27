# Dynamic-object masking

Enable `masking.enabled` in a pipeline profile and set
`masking.weights_path` to a locally staged Torchvision
`maskrcnn_resnet50_fpn_v2` state dictionary. The pipeline then inserts the
`dynamic-mask` stage before preparation and records its artifact at
`data/dynamic_masks/<dataset>/<scene>/` unless `masking.mask_root` overrides
the location.

The generated PNGs are alpha masks: white pixels are static and black pixels
are confirmed moving actors. The artifact also contains `manifest.json` with
the model SHA-256, thresholds, moving-track evidence, and per-frame retention
statistics. Do not alter masks after generation without regenerating the
manifest.

KITTI-360 confirms movement from stereo-derived world tracks. NCore uses its
cuboid tracks for motion and only masks a segmentation instance when a
confirmed cuboid projects into it. Candidates without enough evidence remain
unmasked on purpose; this preserves parked vehicles and avoids treating all
road users as dynamic.

Run a masked scene as a distinct output from the baseline, then compare
held-out views only over static pixels and inspect the overlay/point-cloud
artifacts before adopting it.
