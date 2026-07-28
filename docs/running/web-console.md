# Web Experiment Console

`vbogs-web` is the browser control plane for queued VBOGS experiments. It
uses the existing Torch, JAX, pipeline, render, and artifact volumes; it does
not introduce another training runtime.

## Deploy securely

The Compose profiles publish the service on `127.0.0.1:8090` by default:

```text
VBOGS_WEB_HOST_BIND=127.0.0.1
VBOGS_WEB_HOST_PORT=8090
VBOGS_GUI_GPU_IDS=0,1
VBOGS_AUTH_USER_HEADER=X-Forwarded-User
VBOGS_GUI_ADMINS=alice@example.com
```

Put a TLS/VPN reverse proxy in front of that port. The proxy must remove any
client-supplied identity header and inject the verified user identity in
`X-Forwarded-User` (or the configured equivalent). Requests without that
header receive `401`; do not expose the service directly to the Internet.

`VBOGS_GUI_ADMINS` is a comma-separated allowlist of administrators. If
`VBOGS_GUI_VIEWERS` is empty, every authenticated user is an operator; set it
to a comma-separated allowlist to make all other authenticated users
view-only.

Build the web image alongside the normal stack images:

```bash
docker compose --project-directory . -f docker/compose/compose.yml build vbogs-web
docker compose --project-directory . -f docker/compose/compose.yml up -d vbogs-web
```

## Operating runs

The UI discovers only mounted KITTI-360 drives and converted NCore clips. A
submission writes an immutable resolved configuration and request record under
`data/gui/runs/<run-id>/`; mutable stage artifacts stay below that same run
workspace and curated results go to `outputs/gui/runs/<run-id>/`.

Recipes live in `configs/gui/presets/`. They explicitly list the only fields
accepted from the guided form or advanced YAML panel. Paths, Docker settings,
and upload/credential settings are never accepted from the browser.

Each GPU named by `VBOGS_GUI_GPU_IDS` has one FIFO pipeline slot. Runs can be
cancelled or resumed from a valid later stage; cancellation signals the
recorded process group of the active stage only, not a shared service
container. The **Experiments** page shows only queued and in-progress work.
Completed training results are kept in the separate **Trained runs** catalog,
where operators can browse artifacts, compare completed scenes, and open the
viewer. The active run detail streams pipeline lifecycle events and reads the
run-local log.

Completed runs expose **View scene** as soon as the run has both
`artifacts/train_run.json` and `artifacts/m4/<scene>/U.npy`; a bundle export is
preferred when available but is not required. The full-page viewer is served
inside the authenticated console and keeps the chosen scene on one shared GPU.
Any operator may load or stop a scene. Loading a different run requires an
explicit replacement confirmation and reserves the first available configured
GPU slot.

The console proxies renderer HTTP and WebSocket requests through
`VBOGS_GUI_RENDER_INTERNAL_URL` (default
`http://vbogs-vbgs-render:8070`). The renderer's host port now binds to
`127.0.0.1` by default; set `VBOGS_RENDER_VIEWER_HOST_BIND=0.0.0.0` only for a
separate trusted-network debug viewer.
