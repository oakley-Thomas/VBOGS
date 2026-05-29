# Realtime Viewer

The realtime viewer is a browser-based debug tool for trained Octree-AnyGS
scenes. It runs in the `vbogs-torch` environment, keeps the scene loaded on the
GPU, and streams server-rendered RGB and uncertainty frames to the browser.

This is intended for trusted local debugging, not as an authenticated public
service. Bind it only on networks you trust.

## Start From Docker

Start the local dev stack:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  up -d --no-build
```

Run the viewer:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  exec vbogs-torch \
  python scripts/view_octree_anygs.py \
    --drive 2013_05_28_drive_0007_sync \
    --resolution 4
```

Open:

```text
http://localhost:8070
```

The dev compose overlay maps `${VBOGS_VIEWER_PORT:-8070}` on the host to port
`8070` in the Torch container.

## Useful Modes

Use `--rgb-only` to inspect Octree-AnyGS renderings before `U.npy` exists:

```bash
python scripts/view_octree_anygs.py \
  --drive 2013_05_28_drive_0007_sync \
  --rgb-only
```

Use an explicit model or uncertainty artifact:

```bash
python scripts/view_octree_anygs.py \
  --model-path /data/OCTREE-ANYGS/2013_05_28_drive_0007_sync/<run> \
  --u-path data/m4/2013_05_28_drive_0007_sync/U.npy \
  --resolution 4
```

## Teleport To A Pose

The browser toolbar includes six pose fields:

- `x`, `y`, `z`
- `yaw`, `pitch`, `roll`

You can also paste one full `x y z yaw pitch roll` string into any pose field
to populate all six fields at once.

The CLI and API also accept:

- 12 row-major values for a `3x4` camera-to-world matrix
- 16 row-major values for a `4x4` camera-to-world matrix
- JSON with `position` and `yaw_pitch_roll_deg`, `c2w`, `w2c`, or `matrix`

Angles are degrees, and the Euler convention is:

```text
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
```

The selected viewer camera still supplies intrinsics and image size. Teleport
only changes the camera extrinsics.

Start the viewer at a custom pose:

```bash
python scripts/view_octree_anygs.py \
  --drive 2013_05_28_drive_0007_sync \
  --initial-pose 0 0 2 0 0 0
```

Use a pose file:

```bash
python scripts/view_octree_anygs.py \
  --drive 2013_05_28_drive_0007_sync \
  --initial-pose-file pose.json \
  --initial-pose-convention c2w
```

Render one arbitrary pose through the API:

```bash
curl -X POST http://localhost:8070/api/render \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "test:0",
    "layer": "side_by_side",
    "pose": "0 0 2 0 0 0",
    "quality": 85
  }'
```

The API returns JSON with render metadata and a base64-encoded JPEG:

```json
{
  "metadata": {"camera_id": "test:0", "mode": "side_by_side"},
  "jpeg_base64": "..."
}
```

## Notes

- RGB uses upstream Octree-AnyGS rendering.
- Uncertainty uses `vbogs.render.render_scalar`.
- The viewer drops stale camera updates while a render is in flight, so freefly
  inspection stays responsive even when the GPU is busy.
- If `--rgb-only` is not set, startup validates that `U.npy` has one value per
  Octree-AnyGS anchor.
