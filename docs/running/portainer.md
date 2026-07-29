# Portainer, Uploads, and File Browser

Use `configs/pipeline/portainer.yaml` with `docker/compose/deploy.yml` for
server deployment. The Portainer profile uses compose-managed Docker volumes
for datasets, intermediate artifacts, and curated outputs. None of those
volumes are marked `external`.

## Image Build Choices

For normal Portainer deployment, use `docker/compose/deploy.yml`. It has no
`build:` directives and pulls published images.

If Portainer is attached to the same local Docker environment that runs the
stack, `docker/compose/portainer-build.yml` can build the VBOGS images during
stack deployment. This is not supported for remote Docker environments managed
through the Portainer agent.

For remote environments with shell access, build on the GPU host outside
Portainer, then deploy with `docker/compose/portainer-local.yml`:

```bash
cd /path/to/VBOGS
bash scripts/build_stack_serial.sh
```

Then create the Portainer custom template with:

```text
Compose path: docker/compose/portainer-local.yml
```

That compose file uses `pull_policy: never`, so Portainer relies on the cached
`local/vbogs-*` images already present on the Docker host.

If you only have the Portainer web UI, use one of these paths:

1. Use `docker/compose/deploy.yml` and pull the published registry images.
2. Or build each image from Portainer's **Images > Build a new image** page,
   then deploy `docker/compose/portainer-local.yml`.

For the web-UI build path, build these image names from the public repository
URL `https://github.com/oakley-Thomas/VBOGS.git`:

| Image name | Dockerfile path |
| --- | --- |
| `local/vbogs-torch` | `docker/torch.Dockerfile` |
| `local/vbogs-jax` | `docker/jax.Dockerfile` |
| `local/vbogs-vbgs-render` | `docker/vbgs-render.Dockerfile` |
| `local/vbogs-pipeline` | `docker/pipeline.Dockerfile` |
| `local/vbogs-web` | `docker/web.Dockerfile` |

The service images install runtime dependencies, but they no longer bake a
VBOGS checkout into the image. After the stack is running, open a console in
`vbogs-pipeline` and run:

```bash
vbogs-bootstrap-repo
```

The bootstrap prompts for a GitHub username and token, then fetches VBOGS into
the shared `vbogs-repo` volume mounted by every runtime service. To target a
branch, tag, or commit other than `main`:

```bash
vbogs-bootstrap-repo --ref <branch-tag-or-commit>
```

The token is passed to Git through `GIT_ASKPASS` and is not written into
`.git/config`.

## Server Update Workflow

After bootstrapping the shared checkout, update it from a `vbogs-pipeline`
console:

```bash
python scripts/update_stack_git_ref.py <branch-tag-or-commit>
```

All VBOGS services see the updated checkout through the same Docker volume.

## Container-Side Pipeline Runs

Run stage orchestration from inside `vbogs-pipeline`. The stack mounts the
Docker socket into that container so it can resolve and execute sibling
services by label:

```bash
scripts/run_pipeline.sh \
  --config configs/pipeline/portainer.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after bundle
```

## Google Drive Upload

Enable upload with `--upload-google-drive` or `upload.enabled: true`.

Manual upload example from inside `vbogs-pipeline`:

```bash
python scripts/upload_google_drive.py \
  --config configs/pipeline/portainer.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --folder-id <google-drive-folder-id> \
  --service-account-file /run/secrets/vbogs-google-drive-service-account.json
```

Important upload controls:

| Setting | Meaning |
| --- | --- |
| `VBOGS_GDRIVE_REMOTE` | rclone remote name, default `vbogs_gdrive` |
| `VBOGS_GDRIVE_FOLDER_ID` | Drive folder id used as the upload root |
| `VBOGS_GDRIVE_SERVICE_ACCOUNT_FILE` | JSON credential file path inside container |
| `VBOGS_GDRIVE_SERVICE_ACCOUNT_CREDENTIALS` | Raw JSON credentials from environment/secrets |
| `VBOGS_GDRIVE_DEST` | Destination path inside the remote folder |
| `VBOGS_GDRIVE_SOURCE` | Override upload source |
| `VBOGS_GDRIVE_DRY_RUN=1` | Print upload command without transferring |

Keep service-account JSON in environment/secrets, not in committed configs.

## Realtime Renderer Viewer

The Portainer compose files publish `vbogs-vbgs-render` so the realtime browser
viewer can be reached from another machine. Start the viewer inside the render
container after the Octree-AnyGS scene and `U.npy` exist:

```bash
python scripts/view_octree_anygs.py \
  --drive 2013_05_28_drive_0007_sync \
  --resolution 4
```

Then open:

```text
http://<server-host>:8071
```

The host bind and port are controlled by:

```bash
VBOGS_RENDER_VIEWER_HOST_BIND=0.0.0.0
VBOGS_RENDER_VIEWER_HOST_PORT=8071
```

The viewer is not authenticated. Expose it only on trusted networks or put it
behind your normal VPN, firewall, or TLS reverse proxy.

## File Browser Artifact Access

The Portainer compose files include `vbogs-filebrowser`, a
[File Browser](https://filebrowser.org/installation) sidecar for project files
and generated artifacts. It serves the same stack volumes as the pipeline
containers without mounting the Docker socket. The `/outputs` mount is writable
for uploads; the project, data, generated-config, COLMAP, and Octree-AnyGS
mounts are read-only.

The host port defaults to `8088`:

```text
http://<server-host>:8088
```

Change the bind address or host port with stack variables when needed:

```bash
VBOGS_FILEBROWSER_HOST_BIND=0.0.0.0
VBOGS_FILEBROWSER_HOST_PORT=18088
```

On first boot, File Browser creates the `admin` account and prints the generated
password once in the `vbogs-filebrowser` logs. In Portainer, open the
`vbogs-filebrowser` container logs before logging in. For local compose:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  logs vbogs-filebrowser
```

Useful paths:

| Path | Contents |
| --- | --- |
| `/project` | Read-only VBOGS checkout. In local dev this is the live bind-mounted repo; in Portainer this is the shared bootstrapped repo volume. |
| `/outputs` | Writable run outputs such as `v1_0/<scene-id>/<scene-id>.zip`, unpacked scene folders, and diagnostics. |
| `/data` | Read-only VBOGS data volume, including `KITTI-360` and NVIDIA NCore mounts. |
| `/COLMAP` | Read-only prepared COLMAP-style scene inputs. |
| `/OCTREE-ANYGS` | Read-only Octree-AnyGS checkpoints and training outputs. |
| `/generated_configs` | Read-only generated pipeline and Octree-AnyGS configs. |

After a run reaches `bundle`, download the scene zip from `/outputs`:

| File Browser path | Use |
| --- | --- |
| `/outputs/v1_0/<scene-id>/<scene-id>.zip` | Diagnostics plus the portable local viewer package for rendering and uncertainty API queries. |
| `/outputs/v1_0/<scene-id>/local_viewer/` | Browsable form of the local viewer export included in the zip. |

Because this service can modify the shared output volume, expose it only on
trusted networks or put it behind your normal VPN, firewall, or TLS reverse
proxy. Set `VBOGS_FILEBROWSER_BASE_URL` when the proxy serves it from a subpath.
