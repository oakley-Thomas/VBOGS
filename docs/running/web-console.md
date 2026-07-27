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
container. The run detail page streams pipeline lifecycle events and reads the
run-local log.

Completed runs can be compared in pairs. An administrator can select one
completed portable bundle for the shared realtime viewer; this reserves its
GPU slot and replaces the existing shared viewer session.
