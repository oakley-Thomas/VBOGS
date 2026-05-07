# VBOGS
Combining Octree-GS's scene scalability with Variational Bayes GS uncertainty for better autonomous vehicle mapping

## Environment Notes
The Docker workflow uses one compose stack with three services:

- `vbogs-torch` for Octree-AnyGS, stereo, and bucketing
- `vbogs-jax` for VBGS anchor fitting, fit inspection, and uncertainty scalar
  computation
- `vbogs-pipeline` for running the stages in order inside the stack

### Build commands
```bash
# Rebuild all containers
bash scripts/build_stack_serial.sh

# Rebuild individual containers
bash scripts/build_stack_serial.sh vbogs-torch
bash scripts/build_stack_serial.sh vbogs-jax
bash scripts/build_stack_serial.sh vbogs-pipeline

# Retag local builds and push them to Docker Hub (optional)
bash scripts/publish_dockerhub_images.sh
```

### Docker Volumes:
These external volumes **MUST EXIST** when deploying the stack.
- `KITTI-360`, mounted at `/workspace/VBOGS/data/KITTI-360`
- `COLMAP`, mounted at `/data/COLMAP`
- `OCTREE-ANYGS`, mounted at `/data/OCTREE-ANYGS`
```bash
# Create the volumes (only need to execute once)
docker volume create KITTI-360
docker volume create COLMAP
docker volume create OCTREE-ANYGS
```

### Portainer Deployment
When deploying the stack using Portainer's web interface
1. Create the custom template 
    - Build from: "Repository"
    - Repository URL: https://github.com/oakley-Thomas/VBOGS 
    - Compose path: docker-compose.portainer.yml
2. Create the stack from custom template


### Code updates and Git Repositories
The VBOGS codebase is configured as a Git Repository. If you would like to try out a feature branch, run this script inside of the vbogs-pipeline container to update the stack.
```bash
python scripts/update_stack_git_ref.py <name-of-branch>
```

## Usage

**NOTE:** For development purposes, local Docker Compose automatically reads `docker-compose.override.yml`, which
bind-mounts this checkout at `/workspace/VBOGS` in every service. Edits on your
dev machine are therefore visible inside the running containers without
rebuilding the images.

Start the stack:
```bash
docker compose up -d --no-build
```

Download the KITTI-360 Dataset:
```bash
cd data
./download_kitti_360.sh
```

Run the full pipeline
```bash
# Enter the vbogs-pipeline container
docker compose exec vbogs-pipeline bash

# Run the full pipeline:
python scripts/run_drive_pipeline.py \
  --drive 2013_05_28_drive_0000_sync \
  --gpu 0 \
  --jax-device 0 \
  --use-service-labels
```
**NOTE:** For the full list of pipeline flags, see
[docs/RUN_DRIVE_PIPELINE_ARGS.md](docs/RUN_DRIVE_PIPELINE_ARGS.md).

**NOTE:** For local development, `docker-compose.override.yml` bind-mounts the repo's
`outputs/` directory into the Torch and JAX containers. Final render artifacts
therefore show up directly under `VBOGS/outputs/` on your dev machine.

Run the pipeline from a given step:

**prepare -> train -> stereo -> bucket -> fit -> inspect -> uncertainty -> map-viz -> render**

```bash
python scripts/run_drive_pipeline.py \
  --drive 2013_05_28_drive_0000_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at <start-stage> \
  --stop-after <end-stage> \
  --use-service-labels
```

Development machine verification pipeline run:
```bash
python scripts/run_drive_pipeline.py \
  --drive 2013_05_28_drive_0000_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after render \
  --frame-step 20 \
  --max-frames 30 \
  --resolution 4 \
  --iterations 7000 \
  --max-points-per-frame 50000 \
  --max-observed-anchors 50 \
  --render-max-views 2 \
  --use-service-labels
```


