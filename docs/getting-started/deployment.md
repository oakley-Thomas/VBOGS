# Deployment

## Portainer
To deploy on Portainer, use the prebuilt images on Dockerhub

1. "Add Stack"
2. "Upload" - docker/compose/deploy.yml
3. "Load variables from .env file" - configs/pipeline/portainer.yaml


## Build Images
```bash
bash scripts/build_stack_serial.sh
```

**IMPORTANT NOTE:** by default, ```scripts/build_stack_serial.sh``` will compile ```gsplat``` against the CUDA architecture on the machine that builds the images. If you intend to deploy on a different CUDA architecture, you need to specify the supported versions using ```--cuda-arch-list```.

```bash
# Example - supports RTX Quadro 8000 (sm_7.5), RTX 4070 (sm_8.9), and TYX 5080 (sm_12.0) and 
bash scripts/build_stack_serial.sh --cuda-arch-list '7.5;8.9;12.0'
```

To rebuild one service:
```bash
bash scripts/build_stack_serial.sh vbogs-torch
bash scripts/build_stack_serial.sh vbogs-jax
bash scripts/build_stack_serial.sh vbogs-vbgs-render
bash scripts/build_stack_serial.sh vbogs-pipeline
```
Use `--no-cache` to rebuild from scratch

### Publish to Dockerhub (optional)
To publish the built images to Docker Hub:
```bash
docker login
bash scripts/push_stack_images.sh <dockerhub-username> <version>
```

## Update git stack
```bash
cd /workspace/VBOGS
python scripts/update_stack_git_ref.py <branch-name>
```

