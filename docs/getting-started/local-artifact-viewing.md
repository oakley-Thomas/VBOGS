# Download and View Server Artifacts Locally

This runbook explains how to take a completed pipeline run from a server,
download the portable viewer artifact, open it on a local machine, and query
rendered uncertainty.

Use this page after a pipeline run reaches the `bundle` stage. The normal full
pipeline command writes one zip that contains diagnostics plus the portable
local viewer export.

## Two Bundle Layouts

Two different producers write viewer-ready bundles, and their layouts differ. Check
which one you have before following the steps below.

| Producer | Viewer folder | Model | Cameras | Uncertainty |
| --- | --- | --- | --- | --- |
| Full pipeline (`bundle` stage) | `local_viewer/` | `model/` | `prepared/` | `uncertainty/U.npy` |
| Uncertainty-Evaluation `export` stage | `export/` | `splat/` | `prepared/` | `uncertainty/U.npy` |

The rest of this page covers the pipeline layout. For an Uncertainty-Evaluation run,
read `export/VIEWER_COMMANDS.md` inside the bundle, or see
[the experiment page](../experiments/uncertainty-evaluation.md#interactive-viewer). The
one substantive difference: that experiment holds its test split out of COLMAP entirely,
so its export has no test cameras and the viewer must be started with
`--camera-source train`.

## What to Download

Download the scene bundle zip:

```text
outputs/v1_0/<scene-id>/<scene-id>.zip
```

The zip includes static diagnostics and `local_viewer/`, which contains:

- `model/`: Octree-AnyGS checkpoint and patched config
- `prepared/`: COLMAP-style camera metadata without source image files
- `uncertainty/U.npy`: per-anchor uncertainty values
- `VIEWER_COMMANDS.md`: generated commands for the exported run
- `local_viewer_manifest.json`: source paths and export metadata

## Step 1: Confirm the Bundle Exists on the Server

From inside the server's `vbogs-pipeline` container:

```bash
ls -lh outputs/v1_0/<scene-id>/<scene-id>.zip
ls -lh outputs/v1_0/<scene-id>/local_viewer/
```

If `local_viewer/` is missing but the model, prepared data, and `U.npy` exist,
generate a standalone export manually:

```bash
python scripts/export_local_viewer_run.py \
  --drive <scene-id> \
  --model-path /data/OCTREE-ANYGS/<scene-id>/<run>
```

For pipeline-bundled runs, the export is created automatically unless the run
used `--skip-local-viewer-export`. A bundle created with that flag is
diagnostics-only and cannot be used for local rendering by itself.

## Step 2: Download the Zip

### Option A: File Browser

Open File Browser from your workstation:

```text
http://<server-host>:8088
```

If you need credentials, open a shell in `vbogs-pipeline` on the server:

```bash
python scripts/get_filebrowser_login.py --reset-password
```

In File Browser, download:

```text
/outputs/v1_0/<scene-id>/<scene-id>.zip
```

### Option B: Shell Access

If you can reach the server with SSH, copy the same zip with your normal
server path or volume mount path. For example:

```bash
scp <server-user>@<server-host>:/path/to/outputs/v1_0/<scene-id>/<scene-id>.zip .
```

The exact host path depends on how the Docker volume is mounted. File Browser
is usually simpler for Portainer deployments because it exposes the stack's
`/outputs` volume directly.

## Step 3: Extract It Under the Local Repo

On your local machine, use a checked-out VBOGS repo with the Docker stack built.
Extract the artifact somewhere under the repo, but not under `outputs/`.

The local Docker compose stack mounts `outputs/` as a Docker volume, which can
hide files you extracted into the host checkout. A top-level
`local_viewer_exports/` directory avoids that problem.

```bash
cd /path/to/VBOGS
mkdir -p local_viewer_exports/<scene-id>
unzip /path/to/<scene-id>.zip -d local_viewer_exports/<scene-id>
```

After extraction, this folder should exist:

```text
local_viewer_exports/<scene-id>/<scene-id>/local_viewer/
```

## Step 4: Start the Local Render Container

Start the local development stack:

```bash
./dc_up.sh
```

Start the viewer inside `vbogs-vbgs-render`:

```bash
./dc_bash.sh vbogs-vbgs-render

EXPORT_DIR=/workspace/VBOGS/local_viewer_exports/<scene-id>/<scene-id>/local_viewer

python scripts/view_octree_anygs.py \
  --model-path "${EXPORT_DIR}/model" \
  --u-path "${EXPORT_DIR}/uncertainty/U.npy" \
  --resolution 4
```

Open the browser viewer from the host:

```text
http://localhost:8071
```

The viewer process listens on container port `8070`; the compose stack publishes
it on host port `8071` by default.

## Step 5: Check the Loaded Scene

From the host machine:

```bash
curl http://localhost:8071/api/metadata
curl http://localhost:8071/api/cameras
```

Use the camera IDs returned by `/api/cameras` in the render and uncertainty
queries below. Common IDs look like `test:0` or `train:0`.

## Step 6: Render RGB and Uncertainty Images

Render an RGB image:

```bash
curl -X POST http://localhost:8071/api/render \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "test:0",
    "layer": "rgb",
    "quality": 90
  }'
```

Render an uncertainty heatmap:

```bash
curl -X POST http://localhost:8071/api/render \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "test:0",
    "layer": "uncertainty",
    "quality": 90
  }'
```

The response includes a base64-encoded JPEG in `jpeg_base64`. Use the browser
viewer for quick visual inspection, or decode that field in a script when you
need image files.

## Step 7: Query Numeric Uncertainty

Query the rendered anchors and NBV-style uncertainty score:

```bash
curl -X POST http://localhost:8071/api/rendered-anchors \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "test:0",
    "max_anchors": 25
  }'
```

The response includes:

- `anchors`: visible anchor rows with `anchor_id`, `xyz`, and `uncertainty`
- `uncertainty_image_sum`: integrated rendered uncertainty
- `alpha_sum`: rendered opacity sum
- `alpha_normalized_uncertainty`: the NBV-style score

The score is:

```text
alpha_normalized_uncertainty = uncertainty_image_sum / max(alpha_sum, 1e-8)
```

To query a custom pose, add a pose string to the POST body:

```bash
curl -X POST http://localhost:8071/api/rendered-anchors \
  -H 'Content-Type: application/json' \
  -d '{
    "camera_id": "test:0",
    "pose": "0 0 2 0 0 0",
    "max_anchors": 25
  }'
```

The pose format is:

```text
x y z yaw pitch roll
```

Angles are in degrees. The selected `camera_id` still supplies the camera
intrinsics and image size; the pose only replaces the camera extrinsics.

## Troubleshooting

If the viewer cannot find the model or `U.npy`, check that `EXPORT_DIR` points
at the extracted `local_viewer` folder, not at the zip's parent directory.

If the browser cannot connect, check that the viewer command is still running
inside `vbogs-vbgs-render` and open `http://localhost:8071`, not
`http://localhost:8070`, when using Docker compose.

If `POST /api/rendered-anchors` is unavailable, the viewer was started without
a valid uncertainty file. Restart it with:

```bash
--u-path "${EXPORT_DIR}/uncertainty/U.npy"
```

If files extracted under local `outputs/` do not appear in the container, move
or re-extract the zip under `local_viewer_exports/`. The local compose stack
mounts a Docker volume over `/workspace/VBOGS/outputs`.

## Related Pages

- [Quickstart](index.md)
- [Render Server API](../running/realtime-viewer.md)
- [Data Setup](data.md)
