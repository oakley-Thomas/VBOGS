# Realtime Viewer

The realtime viewer is a browser-based debug tool for trained Octree-AnyGS
scenes. It can run in any Torch-capable VBOGS service, but the stack publishes
the `vbogs-vbgs-render` service for browser access. The viewer keeps the scene
loaded on the GPU and streams server-rendered RGB and uncertainty frames to the
browser.

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
  exec vbogs-vbgs-render \
  python scripts/view_octree_anygs.py \
    --drive 2013_05_28_drive_0007_sync \
    --resolution 4
```

Open:

```text
http://localhost:8071
```

The compose stack maps
`${VBOGS_RENDER_VIEWER_HOST_BIND:-127.0.0.1}:${VBOGS_RENDER_VIEWER_HOST_PORT:-8071}`
on the host to port `8070` in `vbogs-vbgs-render`. For a remote Portainer
server, open:

```text
http://<server-host>:8071
```

Change the host port with `VBOGS_RENDER_VIEWER_HOST_PORT` if `8071` is already
in use. The local dev overlay still maps `${VBOGS_VIEWER_PORT:-8070}` to
`vbogs-torch` for older Torch-container viewer workflows.

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

## Viewer API

The viewer exposes local REST endpoints while `scripts/view_octree_anygs.py` is
running. These APIs use the already-loaded Octree-AnyGS scene, so they avoid
reloading the model for every query.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/metadata` | Viewer/session metadata, including drive, model path, render modes, and uncertainty scale |
| `GET /api/cameras` | Train/test camera list with intrinsics and camera-to-world matrices |
| `POST /api/render` | Render RGB, uncertainty, alpha, or side-by-side JPEG from a selected or custom pose |
| `POST /api/rendered-anchors` | Return rendered parent anchors for a view, with per-anchor uncertainty and aggregate uncertainty totals |

All `POST` endpoints accept `camera_id`; if omitted, the viewer default camera
is used. Custom pose fields are optional and use the same conventions described
above:

- `pose`: `x y z yaw pitch roll`, 12 matrix values, or 16 matrix values
- `c2w`: camera-to-world matrix
- `w2c`: world-to-camera matrix
- `matrix` plus optional `pose_convention`
- `position` plus `yaw_pitch_roll_deg`

Available `/api/render` layers:

- `rgb`: upstream Octree-AnyGS RGB render
- `uncertainty`: visual uncertainty heatmap JPEG
- `alpha`: visibility/opacity image
- `side_by_side`: RGB next to uncertainty heatmap

The image endpoint returns JPEGs for inspection and client display. Use
`/api/rendered-anchors` when you need numeric uncertainty totals or NBV scores.

### Render RGB or Uncertainty Images

Render an RGB image:

```bash
curl -X POST http://localhost:8071/api/render \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "test:0",
    "layer": "rgb",
    "pose": "0 0 2 0 0 0",
    "quality": 85
  }'
```

Render an uncertainty heatmap image:

```bash
curl -X POST http://localhost:8071/api/render \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "test:0",
    "layer": "uncertainty",
    "pose": "0 0 2 0 0 0",
    "quality": 85
  }'
```

The API returns JSON with render metadata and a base64-encoded JPEG:

```json
{
  "metadata": {"camera_id": "test:0", "mode": "rgb"},
  "jpeg_base64": "..."
}
```

### Query Uncertainty Score

Query rendered anchors, integrated uncertainty, and the alpha-normalized score:

```bash
curl -X POST http://localhost:8071/api/rendered-anchors \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "test:0",
    "pose": "0 0 2 0 0 0",
    "max_anchors": 25
  }'
```

The response includes:

- `anchors`: unique rendered parent anchors with `anchor_id`, `xyz`,
  `uncertainty`, optional `level`, and `rendered_gaussian_count`
- `total_anchor_uncertainty`: sum of the full rendered anchor set's `U` values
- `uncertainty_image_sum`: integrated rendered uncertainty image value
- `alpha_sum` and `alpha_normalized_uncertainty`

The NBV-style score is:

```text
alpha_normalized_uncertainty = uncertainty_image_sum / max(alpha_sum, 1e-8)
```

Use `max_anchors` in the request body to limit how many anchor rows are returned
while still computing totals over the full rendered-anchor set.

`POST /api/rendered-anchors` requires the viewer to be started with a valid
`U.npy`; it is unavailable in `--rgb-only` mode.

## Example Scripts

### Capture API Views

Example script: `scripts/capture_viewer_api_views.py`

Capture the first few training views through the API:

```bash
python scripts/capture_viewer_api_views.py \
  --base-url http://localhost:8071 \
  --source train \
  --count 5 \
  --layer side_by_side
```

By default this writes JPEGs, per-image sidecars, rendered-anchor JSON files,
and `capture_manifest.json` under
`outputs/viewer_api_captures/<drive>/train/side_by_side/`.

Use `--max-rendered-anchors N` to cap anchor rows per view while preserving
the full uncertainty totals, or `--skip-rendered-anchors` to capture only the
rendered images.

If you are running from the Docker dev stack, `outputs/` is a Docker volume, not
a host checkout directory. Download captures through File Browser at
`/outputs`, or copy them back to the host with:

```bash
mkdir -p outputs/viewer_api_captures

docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  cp vbogs-vbgs-render:/workspace/VBOGS/outputs/viewer_api_captures/<drive> \
     outputs/viewer_api_captures/
```

## Notes

- RGB uses upstream Octree-AnyGS rendering.
- Uncertainty uses `vbogs.render.render_scalar`.
- The viewer drops stale camera updates while a render is in flight, so freefly
  inspection stays responsive even when the GPU is busy.
- If `--rgb-only` is not set, startup validates that `U.npy` has one value per
  Octree-AnyGS anchor.
