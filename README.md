# VBOGS
Combining Octree-GS's scene scalability with Variational Bayes GS uncertainty for better autonomous vehicle mapping

## Environment Notes

VBOGS uses two separate conda environments because the PyTorch and JAX CUDA stacks
conflict in practice:

- `vbogs-torch` for Octree-AnyGS, stereo, point bucketing, and rendering
- `vbogs-jax` for VBGS fitting and posterior computations

The repo helper script is [scripts/envs.sh](/home/oakley/ub/advanced_robotics/VBOGS/scripts/envs.sh:1).

Common commands:

```bash
bash scripts/envs.sh create-torch
bash scripts/envs.sh create-jax
bash scripts/envs.sh check-torch-stack
bash scripts/envs.sh smoke-test-jax
```

### Torch Stack

The current `vbogs-torch` setup is intentionally pinned to a CUDA 12.8 PyTorch
wheel stack:

- Python `3.10`
- PyTorch `2.7.1+cu128`
- torchvision `0.22.1+cu128`
- torchaudio `2.7.1+cu128`
- `torch_scatter` wheel matched to `torch 2.7 / cu128`
- `gsplat` installed from the official `pt27cu128` wheel index

This configuration is chosen because it works on the local RTX 5080 dev machine
and is a reasonable deployment target for the Quadro RTX 8000 server, assuming
the server's NVIDIA driver is new enough for CUDA 12.8-era PyTorch wheels.

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
