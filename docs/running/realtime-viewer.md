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

## Notes

- RGB uses upstream Octree-AnyGS rendering.
- Uncertainty uses `vbogs.render.render_scalar`.
- The viewer drops stale camera updates while a render is in flight, so freefly
  inspection stays responsive even when the GPU is busy.
- If `--rgb-only` is not set, startup validates that `U.npy` has one value per
  Octree-AnyGS anchor.
