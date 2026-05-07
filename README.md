# VBOGS
Combining Octree-GS's scene scalability with Variational Bayes GS uncertainty for better autonomous vehicle mapping

## Environment Notes

Common commands:

```bash
bash scripts/envs.sh create-torch
bash scripts/envs.sh create-jax
bash scripts/envs.sh check-torch-stack
bash scripts/envs.sh smoke-test-jax
```

## Deployment Quick Start

The Docker workflow uses one compose stack with three services:

- `vbogs-torch` for Octree-AnyGS, stereo, and bucketing
- `vbogs-jax` for VBGS anchor fitting, fit inspection, and uncertainty scalar
  computation
- `vbogs-pipeline` for running the stages in order inside the stack

You can run the same `docker-compose.yml` locally with Docker Compose, or deploy
it remotely through the Portainer web UI. The default pipeline still stops
after the M4b fit-inspection helper to force a manual posterior sanity check,
but M5 uncertainty computation and M7 side-by-side diagnostic rendering now have
repo-owned entry points.

For local Docker Compose, see `Local Docker Compose` below. For a remote server
where you only have Portainer web access, see `Remote Portainer Web UI`.

## M2 Training Workflow

M2 now has a repo-owned local workflow for preparing KITTI-360 drive
`2013_05_28_drive_0008_sync` into the COLMAP-style ingest that Octree-AnyGS
expects and then launching a conservative LoD training run that fits inside a
16 GB dev GPU budget.

The repo's source KITTI-360 layout is now expected under:

- `data/KITTI-360/data_2d_test/`
- `data/KITTI-360/data_poses/`
- `data/KITTI-360/calibration/`

The dataset-prep and stereo-export scripts auto-detect that layout by default.
They also still accept `--raw-root`, `--poses-root`, and `--calibration-dir`
overrides if your server stores the source data somewhere else.

To bootstrap that tree from official KITTI/KITTI-360 downloads, fill in the
URLs in [scripts/kitti360_download_manifest.example.json](/home/oakley/ub/advanced_robotics/VBOGS/scripts/kitti360_download_manifest.example.json),
copy it to `data/KITTI-360/download_manifest.json`, and run:

```bash
python scripts/download_kitti360.py --manifest data/KITTI-360/download_manifest.json
```

The downloader writes into `data/KITTI-360/`, verifies expected output paths,
and supports both zip and tar-style archives.

Prepare the dataset:

```bash
python scripts/prepare_kitti360_colmap.py \
  --drive 2013_05_28_drive_0008_sync \
  --frame-step 10 \
  --max-frames 160
```

That writes a prepared dataset under
`/data/COLMAP/2013_05_28_drive_0008_sync/` with:

- `images/`
- `sparse/0/cameras.txt`
- `sparse/0/images.txt`
- `sparse/0/points3D.ply`

Generate a 16 GB-safe config and launch training:

```bash
python scripts/train_octree_anygs.py \
  --dataset-path /data/COLMAP/2013_05_28_drive_0008_sync \
  --gpu 0
```

By default, training runs are written under
`/data/OCTREE-ANYGS/2013_05_28_drive_0008_sync/<timestamp>/`.

The default local preset intentionally trades fidelity for safety:

- `resolution: 4`
- `feat_dim: 16`
- `base_layer: 9`
- `iterations: 15000`
- `render_mode: RGB`
- `add_prefilter: false`
- `densification: false`

If you have headroom after a first successful run, the least risky upgrades are
to lower `--resolution` from `4` to `2` and increase `--iterations`.

The local preset also disables Octree-AnyGS densification because the current
upstream stats path is incompatible with the installed `gsplat` tensor shapes
on this machine. That keeps M2 stable on the dev box at the cost of some final
scene quality.

Use `--write-config-only` to inspect the generated YAML without starting
training.

## M3 Point Cloud Export

M3 exports a dense-ish world-frame stereo point cloud from the same KITTI-360
drive using the `vbogs-torch` env:

```bash
bash -lc 'source scripts/envs.sh activate-torch >/dev/null && \
python scripts/stereo_to_pointcloud.py \
  --drive 2013_05_28_drive_0008_sync \
  --selection-metadata /data/COLMAP/2013_05_28_drive_0008_sync/metadata.json \
  --write-ply'
```

That writes artifacts under `data/points_world/2013_05_28_drive_0008_sync/`:

- `points_world.npz` with keys `xyz`, `rgb`, and `frame_id`
- `points_world_metadata.json` with the matcher and filtering settings
- `points_world.ply` when `--write-ply` is passed for quick viewer sanity checks

The current implementation ships with an `sgbm` provider and a forward-looking
`--matcher` interface so a future RAFT-Stereo backend can preserve the same
output contract. The validity mask keeps only pixels that pass:

- minimum disparity / depth bounds
- left-right consistency
- a local grayscale texture threshold

Use `--pixel-step` and `--max-points-per-frame` to trade off density vs runtime
and file size on the dev machine.

If you need to point at a non-default KITTI-360 checkout on a server, pass the
same input-root overrides here as in M2:

```bash
python scripts/stereo_to_pointcloud.py \
  --raw-root /path/to/KITTI-360/data_2d_test \
  --poses-root /path/to/KITTI-360/data_poses \
  --calibration-dir /path/to/KITTI-360/calibration \
  ...
```

### Torch Stack

The current `vbogs-torch` setup is intentionally pinned to a CUDA 12.8 PyTorch
wheel stack:

- Python `3.10`
- PyTorch `2.7.1+cu128`
- torchvision `0.22.1+cu128`
- torchaudio `2.7.1+cu128`
- `torch_scatter` wheel matched to `torch 2.7 / cu128`
- `gsplat` built from source, because upstream prebuilt wheels do not currently
  cover the `torch 2.7 / cu128 / Python 3.10` combination

This configuration is chosen because it works on the local RTX 5080 dev machine
and is a reasonable deployment target for the Quadro RTX 8000 server, assuming
the server's NVIDIA driver is new enough for CUDA 12.8-era PyTorch wheels.

The Docker build keeps the `gsplat` compile intentionally small:

- `VBOGS_TORCH_MAX_JOBS=1` by default
- `VBOGS_TORCH_CUDA_ARCH_LIST=7.5;12.0` by default in Compose
- `scripts/build_stack_serial.sh` detects GPU 0 and narrows
  `VBOGS_TORCH_CUDA_ARCH_LIST` to that single compute capability unless you
  set it yourself

For the local RTX 5080, use `12.0`. For the Quadro RTX 8000 server, use `7.5`.
Building every architecture is much more likely to exhaust WSL memory.

### Validation

Use the following command after provisioning `vbogs-torch`:

```bash
bash scripts/envs.sh check-torch-stack
```

It verifies:

- CUDA visibility in PyTorch
- a real CUDA tensor operation
- `torch_scatter` CUDA execution
- `gsplat` import
- `gaussian_renderer.render` import through `Octree-AnyGS`

## M4 Point Bucketing And Anchor Fits

M4 is split into:

- `M4a` in `vbogs-torch`: bucket each stereo point into every Octree-AnyGS
  anchor cell that contains it, while also writing the globally normalized
  `(xyz, rgb)` rows that VBGS expects.
- `M4b` in `vbogs-jax`: fit a post-hoc VBGS posterior per anchor from those
  packed point assignments.

Run M4a:

```bash
bash -lc 'source scripts/envs.sh activate-torch >/dev/null && \
python scripts/bucket_points.py \
  --drive 2013_05_28_drive_0008_sync'
```

This writes to `data/m4/2013_05_28_drive_0008_sync/`:

- `points_norm.npz` with `points_norm` plus raw sidecar arrays
- `pts_by_anchor.npz` with packed `anchor_offsets` / `point_indices`
- `norm_params.json`
- `bucket_metadata.json`

The current dev-scene summary on the bundled artifacts is:

- `12,792,935` stereo points
- `267,830` anchors across `9` levels
- `104,577` anchors with at least `20` assigned points

Run a smoke test for M4b:

```bash
bash -lc 'source scripts/envs.sh activate-jax >/dev/null && \
python scripts/fit_anchors.py \
  --drive 2013_05_28_drive_0008_sync \
  --max-observed-anchors 5 \
  --log-every 1'
```

Smoke runs write:

- `anchor_posterior.smoke.npz`
- `fit_metadata.smoke.json`

Run the full M4b fit by omitting `--max-observed-anchors`; that writes:

- `anchor_posterior.npz`
- `fit_metadata.json`

Dense anchors can dominate M4b runtime. Direct `fit_anchors.py` runs are exact
by default, but you can bound dense-anchor work with deterministic random
subsampling:

```bash
python scripts/fit_anchors.py \
  --drive 2013_05_28_drive_0008_sync \
  --max-points-per-anchor 10000
```

The pipeline wrapper defaults to this `10000`-point cap; pass
`--max-points-per-anchor 0` for an exact unbounded pipeline fit.

The initial hyperparameter defaults match `PLAN.md`:

- `K_INIT=10`
- `K_MAX=40`
- `K_GROWTH_FACTOR=2`
- `MIN_POINTS_PER_ANCHOR=20`
- `ELBO_IMPROVEMENT_TOL=0.01`

## Drive Pipeline Runner

The repo includes a stack-contained orchestrator for the implemented pipeline
stages: M2 dataset preparation/training, M3 stereo export, M4a anchor bucketing,
M4b VBGS fitting, and the M4b fit-inspection summary. The compose stack has a
third service, `vbogs-pipeline`, that dispatches each stage into the correct
sibling container:

- `vbogs-torch`: prepare, train, stereo, bucket
- `vbogs-jax`: fit, inspect
- `vbogs-pipeline`: orchestration only

In Portainer, set these stack environment variables and redeploy:

```text
VBOGS_DRIVE=2013_05_28_drive_0008_sync
VBOGS_PIPELINE_AUTORUN=1
VBOGS_PIPELINE_ARGS=--gpu 0 --jax-device 0
```

The pipeline service uses Docker Compose labels to find the running
`vbogs-torch` and `vbogs-jax` containers, then runs the stage commands inside
them with `docker exec`. The data boundary remains the shared mounted volumes:

- `/data/COLMAP/<drive>` for the prepared Octree-AnyGS dataset
- `/data/OCTREE-ANYGS/<drive>/<timestamp>` for the trained model
- `data/points_world/<drive>` for M3 point clouds
- `data/m4/<drive>` for M4a/M4b artifacts

For a short M4b smoke run instead of a full fit, set:

```text
VBOGS_PIPELINE_ARGS=--gpu 0 --jax-device 0 --max-observed-anchors 5 --log-every 1
```

To resume from a later stage after replacing or fixing an artifact, set:

```text
VBOGS_PIPELINE_ARGS=--gpu 0 --jax-device 0 --start-at bucket
```

For a command audit without running the expensive stages:

```text
VBOGS_PIPELINE_ARGS=--gpu 0 --jax-device 0 --dry-run
```

When `VBOGS_PIPELINE_AUTORUN=0`, the pipeline service stays idle with
`sleep infinity`. That is useful for keeping the service in the stack without
accidentally relaunching a long training run on every redeploy.

The pipeline service mounts `/var/run/docker.sock` so it can run commands in
the sibling containers. This is what makes the workflow fully stack-contained,
but it also means the service has Docker daemon access on the host.

This runner defaults to stopping after the M4b fit-inspection summary so you can
perform the required manual validation before M5. To continue from a completed
fit into uncertainty computation and side-by-side rendering, run with:

```text
VBOGS_PIPELINE_ARGS=--gpu 0 --jax-device 0 --start-at uncertainty --stop-after render --render-max-views 1
```

Remove `--render-max-views 1` after the smoke render succeeds.

## Docker Compose And Portainer Deployment

The compose stack is designed to be self-contained. `vbogs-pipeline` is a small
orchestrator container that mounts `/var/run/docker.sock`, finds the sibling
`vbogs-torch` and `vbogs-jax` containers by Compose labels, and runs each stage
inside the correct runtime container. You do not need to manually run stage
commands inside the Torch or JAX containers.

The stack uses these services:

- `vbogs-torch` for M2, M3, and M4a
- `vbogs-jax` for M4b fitting and inspection
- `vbogs-pipeline` for stack-contained orchestration

The stack uses these Docker volumes:

- `KITTI-360`, external, mounted at `/workspace/VBOGS/data/KITTI-360`
- `COLMAP`, external, mounted at `/data/COLMAP`
- `OCTREE-ANYGS`, external, mounted at `/data/OCTREE-ANYGS`
- `vbogs-data`, compose-managed, mounted at `/workspace/VBOGS/data`
- `vbogs-outputs`, compose-managed, mounted at `/workspace/VBOGS/outputs`
- `vbogs-generated-configs`, compose-managed, mounted at `/workspace/VBOGS/generated_configs`

`KITTI-360`, `COLMAP`, and `OCTREE-ANYGS` are declared as external volumes, so
they must exist before the stack starts. The source KITTI-360 volume must contain
the expected layout:

- `/workspace/VBOGS/data/KITTI-360/data_2d_test/`
- `/workspace/VBOGS/data/KITTI-360/data_poses/`
- `/workspace/VBOGS/data/KITTI-360/calibration/`

The pipeline currently supports these stages:

1. `prepare`: KITTI-360 to Octree-AnyGS COLMAP-style dataset
2. `train`: Octree-AnyGS training
3. `stereo`: world-frame stereo point cloud
4. `bucket`: point-to-anchor bucketing
5. `fit`: per-anchor VBGS posterior fitting
6. `inspect`: noninteractive fit summary and anchor candidates for manual review
7. `uncertainty`: M5 posterior-to-scalar reduction, writing `U.npy`
8. `render`: RGB plus uncertainty-map side-by-side diagnostic images

### Top-Level Pipeline Config

Main pipeline knobs live in [pipeline_config.yaml](pipeline_config.yaml). The
runner loads this file by default and uses it for values such as:

- drive id and Git ref
- stage range, dry-run mode, and resume point
- KITTI-360 input overrides
- dataset-prep sampling
- Octree-AnyGS training budget
- stereo density/filtering
- anchor checkpoint selection
- VBGS fit mode, device, batching, and smoke-run cap
- M4b inspection output size and optional explicit anchor export

Values from `VBOGS_PIPELINE_ARGS` or direct CLI flags override the YAML config.
In Compose, set `VBOGS_PIPELINE_CONFIG` if you want to use a different config
path:

```text
VBOGS_PIPELINE_CONFIG=pipeline_config.yaml
```

`VBOGS_DRIVE` is also an override for `pipeline.drive`; omit it if the config
file should be the only source of the drive id.

For Portainer web-only deployments, the usual pattern is to commit a tuned
`pipeline_config.yaml` on a branch, set `VBOGS_GIT_REF` to that branch, and
redeploy the stack.

### Local Docker Compose

Use this when you have terminal access on the machine running Docker.
`docker-compose.yml` bind-mounts the current checkout over `/workspace/VBOGS`,
so local edits replace the copy of VBOGS that was baked into the images.
Initialize the local submodules before starting the stack, because the bind
mount also replaces the baked `Octree-AnyGS` and `vbgs` directories.

Create the required external volumes once:

```bash
docker volume create KITTI-360
docker volume create COLMAP
docker volume create OCTREE-ANYGS
```

Put the KITTI-360 data into the `KITTI-360` volume, or temporarily edit
`docker-compose.yml` to bind-mount your local dataset directory to
`/workspace/VBOGS/data/KITTI-360`. Named volumes are the better match for the
remote Portainer workflow.

One local copy pattern is:

```bash
docker run --rm \
  -v KITTI-360:/dst \
  -v /path/to/KITTI-360:/src:ro \
  alpine sh -c "cp -a /src/. /dst/"
```

Build and start the idle stack:

```bash
VBOGS_TORCH_IMAGE=local/vbogs-torch \
VBOGS_JAX_IMAGE=local/vbogs-jax \
VBOGS_PIPELINE_IMAGE=local/vbogs-pipeline \
VBOGS_TORCH_CUDA_ARCH_LIST=12.0 \
VBOGS_TORCH_MAX_JOBS=1 \
bash scripts/build_stack_serial.sh

VBOGS_TORCH_IMAGE=local/vbogs-torch \
VBOGS_JAX_IMAGE=local/vbogs-jax \
VBOGS_PIPELINE_IMAGE=local/vbogs-pipeline \
docker compose up -d --no-build
```

On WSL, prefer the serial build helper over `docker compose up --build`.
The Torch image compiles `gsplat` while the JAX image downloads large CUDA
wheels; allowing those phases to overlap can exhaust WSL memory or Docker disk
space and crash the VM.

Run a command audit first. This prints the stage commands without launching the
expensive work:

```bash
VBOGS_TORCH_IMAGE=local/vbogs-torch \
VBOGS_JAX_IMAGE=local/vbogs-jax \
VBOGS_PIPELINE_IMAGE=local/vbogs-pipeline \
VBOGS_DRIVE=2013_05_28_drive_0008_sync \
VBOGS_PIPELINE_AUTORUN=1 \
VBOGS_PIPELINE_CONFIG=pipeline_config.yaml \
VBOGS_PIPELINE_GIT_REF= \
VBOGS_PIPELINE_ARGS="--gpu 0 --jax-device 0 --dry-run" \
docker compose up --no-build vbogs-pipeline
```

Run a short smoke fit:

```bash
VBOGS_TORCH_IMAGE=local/vbogs-torch \
VBOGS_JAX_IMAGE=local/vbogs-jax \
VBOGS_PIPELINE_IMAGE=local/vbogs-pipeline \
VBOGS_DRIVE=2013_05_28_drive_0008_sync \
VBOGS_PIPELINE_AUTORUN=1 \
VBOGS_PIPELINE_CONFIG=pipeline_config.yaml \
VBOGS_PIPELINE_GIT_REF= \
VBOGS_PIPELINE_ARGS="--gpu 0 --jax-device 0 --max-observed-anchors 5 --log-every 1" \
docker compose up --no-build vbogs-pipeline
```

Run the full implemented pipeline:

```bash
VBOGS_TORCH_IMAGE=local/vbogs-torch \
VBOGS_JAX_IMAGE=local/vbogs-jax \
VBOGS_PIPELINE_IMAGE=local/vbogs-pipeline \
VBOGS_DRIVE=2013_05_28_drive_0008_sync \
VBOGS_PIPELINE_AUTORUN=1 \
VBOGS_PIPELINE_CONFIG=pipeline_config.yaml \
VBOGS_PIPELINE_GIT_REF= \
VBOGS_PIPELINE_ARGS="--gpu 0 --jax-device 0" \
docker compose up --no-build vbogs-pipeline
```

Local Compose runs the source from your bind-mounted checkout. Leave
`VBOGS_PIPELINE_GIT_REF` empty for normal development so the containers do not
try to change the branch of your working tree.

Resume from a later stage by changing only the pipeline args:

```bash
VBOGS_PIPELINE_ARGS="--gpu 0 --jax-device 0 --start-at bucket"
```

After a one-shot pipeline run, set `VBOGS_PIPELINE_AUTORUN=0` or start the stack
with no pipeline env overrides so redeploys do not relaunch training.

Useful local checks:

```bash
docker compose logs -f vbogs-pipeline
docker compose exec vbogs-torch python scripts/check_torch_stack.py --repo-root /workspace/VBOGS
docker compose exec vbogs-jax python -c "import jax; print(jax.devices())"
```

### Remote Portainer Web UI

Use this when the remote GPU server is only available through the Portainer web
GUI and you cannot run host terminal commands.

1. In Portainer, create these volumes from `Volumes > Add volume`:
   - `KITTI-360`
   - `COLMAP`
   - `OCTREE-ANYGS`

2. Make sure the `KITTI-360` volume already contains the KITTI-360 source tree.
   If your server administrator preloaded a different volume name, either rename
   the volume in Portainer or edit the compose file's external volume name before
   deploying. With only the Portainer web UI, assume the dataset must be
   preloaded by an administrator, a storage plugin, or another managed transfer
   path before the pipeline can run.

   The prep stage supports two common source layouts inside the mounted volume:

   - newer layout:
     `/workspace/VBOGS/data/KITTI-360/data_2d_test/`,
     `/workspace/VBOGS/data/KITTI-360/data_poses/`,
     `/workspace/VBOGS/data/KITTI-360/calibration/`
   - older layout:
     `/workspace/VBOGS/data/KITTI-360/images/`,
     `/workspace/VBOGS/data/KITTI-360/data_poses/`,
     `/workspace/VBOGS/data/KITTI-360/calibration/`

   If your volume uses the older `images/` layout, pass explicit input-root
   overrides through `VBOGS_PIPELINE_ARGS`. The pipeline container itself does
   not mount the KITTI-360 volume, so verify the paths from `vbogs-torch` if
   you need to inspect the source tree in Portainer.

3. In Portainer, create a stack from Git or the Web editor using
   `docker-compose.portainer.yml`.

4. In the stack environment variables, set image names. For a server-side build
   from the Git ref in the Dockerfiles, local names are fine:

```text
VBOGS_TORCH_IMAGE=local/vbogs-torch
VBOGS_JAX_IMAGE=local/vbogs-jax
VBOGS_PIPELINE_IMAGE=local/vbogs-pipeline
VBOGS_GIT_REF=main
```

If Portainer should pull prebuilt registry images instead, set those image names
instead and configure registry credentials in Portainer if the images are
private:

```text
VBOGS_TORCH_IMAGE=ghcr.io/oakley-thomas/vbogs-torch:latest
VBOGS_JAX_IMAGE=ghcr.io/oakley-thomas/vbogs-jax:latest
VBOGS_PIPELINE_IMAGE=ghcr.io/oakley-thomas/vbogs-pipeline:latest
VBOGS_GIT_REF=main
```

In Portainer, `VBOGS_GIT_REF` is also the runtime source ref. On container
startup each service fetches from `VBOGS_GIT_URL`, checks out that branch, tag,
or commit, and updates submodules. Set `VBOGS_GIT_URL` only when the branch
lives in a fork.

5. Deploy the stack with the pipeline idle:

```text
VBOGS_PIPELINE_AUTORUN=0
VBOGS_DRIVE=2013_05_28_drive_0008_sync
VBOGS_PIPELINE_CONFIG=pipeline_config.yaml
VBOGS_PIPELINE_ARGS=
```

6. Check that `vbogs-torch`, `vbogs-jax`, and `vbogs-pipeline` are running.
   Use Portainer's container logs for errors. The idle pipeline logs should say
   it is waiting for `VBOGS_PIPELINE_AUTORUN=1`.

7. For a dry-run command audit, edit the stack environment variables and
   redeploy:

```text
VBOGS_PIPELINE_AUTORUN=1
VBOGS_PIPELINE_CONFIG=pipeline_config.yaml
VBOGS_PIPELINE_ARGS=--gpu 0 --jax-device 0 --dry-run
```

8. For a short smoke fit, redeploy with:

```text
VBOGS_PIPELINE_AUTORUN=1
VBOGS_PIPELINE_CONFIG=pipeline_config.yaml
VBOGS_PIPELINE_ARGS=--gpu 0 --jax-device 0 --max-observed-anchors 5 --log-every 1
```

9. For the full implemented pipeline, redeploy with:

```text
VBOGS_PIPELINE_AUTORUN=1
VBOGS_PIPELINE_CONFIG=pipeline_config.yaml
VBOGS_PIPELINE_ARGS=--gpu 0 --jax-device 0
```

10. Watch `vbogs-pipeline` logs in Portainer. The orchestrator will print each
    stage and the `docker exec` command it runs in the sibling container.

11. After the run finishes, set `VBOGS_PIPELINE_AUTORUN=0` and redeploy so later
    stack updates do not restart the expensive pipeline.

To resume from a later stage through the web UI:

```text
VBOGS_PIPELINE_ARGS=--gpu 0 --jax-device 0 --start-at bucket
```

`VBOGS_PIPELINE_GIT_REF` remains available as an advanced override when the
pipeline run should use a different ref than the rest of the stack. Most
Portainer deployments should leave it unset and use `VBOGS_GIT_REF`.

The pipeline service requires `/var/run/docker.sock` access. That is what makes
the web-only Portainer workflow possible, but it gives `vbogs-pipeline` Docker
daemon access on the host. If your Portainer policy blocks Docker socket mounts,
this orchestration style will not work without a different scheduler.

If the GPU is not visible inside `vbogs-torch` or `vbogs-jax`, check the
container logs and confirm the Portainer host has the NVIDIA container runtime
configured. The compose file requests all GPUs for the Torch and JAX services.

### Deployment Notes

- The source is cloned during image build, so image builds only see committed
  and pushed code.
- M4b depends on `data/m4/<drive>/points_norm.npz` and
  `data/m4/<drive>/pts_by_anchor.npz`, produced by M4a in the same pipeline.
- The current M4b implementation is serial or batched over observed anchors, so
  the GPU is most useful for making the JAX fit stable and fast.
- `anchor_posterior.npz` and `fit_metadata.json` are written to
  `data/m4/<drive>/` and remain available to both runtime containers through the
  shared `vbogs-data` volume.
- M5 writes `U.npy`, `uncertainty_components.npz`,
  `uncertainty_metadata.json`, and `uncertainty_histogram.png` to
  `data/m4/<drive>/`.
- The render stage consumes `U.npy` in `vbogs-torch` and writes images under
  `outputs/uncertainty_views/<drive>/`.
