# Portainer, Uploads, and Transfers

Use `configs/pipeline/portainer.yaml` with `docker/compose/portainer.yml` for
server deployment. The Portainer profile writes outputs to the stack-managed
`vbogs-outputs` volume at `/workspace/VBOGS/outputs`.

## Server Update Workflow

After pulling new code on the GPU server, prefer the repo-owned update path:

```bash
bash scripts/update_server_stack.sh
```

This keeps server rebuilds repeatable instead of relying on ad hoc edits inside
running containers.

## Autorun from Environment

The `vbogs-pipeline` service can launch the drive pipeline automatically:

```bash
VBOGS_PIPELINE_AUTORUN=1
VBOGS_PIPELINE_CONFIG=configs/pipeline/portainer.yaml
VBOGS_DRIVE=2013_05_28_drive_0007_sync
VBOGS_PIPELINE_ARGS="--gpu 0 --jax-device 0 --start-at prepare --stop-after bundle"
```

Internally this calls:

```bash
python scripts/run_pipeline_from_env.py
```

## Google Drive Upload

Enable upload with `--upload-google-drive`, `upload.enabled: true`, or
`VBOGS_GDRIVE_UPLOAD=1`.

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

## SSH/SFTP Artifact Transfer

The Portainer compose file includes `vbogs-transfer`, a read-only SSH/SFTP
service for pulling artifacts from server volumes.

Create a local key:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/vbogs_portainer
```

Set this Portainer stack variable and redeploy:

```bash
VBOGS_TRANSFER_AUTHORIZED_KEYS=<contents-of-~/.ssh/vbogs_portainer.pub>
```

The host port defaults to `22022`. To change it:

```bash
VBOGS_TRANSFER_HOST_PORT=32222
```

Download the curated run zip:

```bash
scp -i ~/.ssh/vbogs_portainer -P 22022 \
  vbogs@<server-host>:/workspace/VBOGS/outputs/v1_0/<drive>.zip .
```

Mirror a full output directory:

```bash
rsync -avP -e "ssh -i ~/.ssh/vbogs_portainer -p 22022" \
  vbogs@<server-host>:/workspace/VBOGS/outputs/v1_0/<drive>/ ./vbogs-run/
```

Disable transfer access by removing `VBOGS_TRANSFER_AUTHORIZED_KEYS` and
redeploying the stack.
