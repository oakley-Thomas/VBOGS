# Testing

Run the Python test suite from the repo root:

```bash
pytest
```

Focused tests:

```bash
pytest tests/test_run_drive_pipeline.py
pytest tests/test_bucket_points.py
pytest tests/test_compute_uncertainty.py
pytest tests/test_render.py
pytest tests/test_query_uncertainty_view.py
pytest tests/test_bundle_run_outputs.py
```

## Docker Smoke Checks

Torch stack:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  exec vbogs-torch python scripts/check_torch_stack.py
```

JAX import check:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  exec vbogs-jax python -c "from vbgs.model.train import fit_gmm_step; print(fit_gmm_step.__name__)"
```

Pipeline dry run:

```bash
scripts/run_pipeline.sh \
  --config configs/pipeline/dev.yaml \
  --dry-run
```

## Docs Checks

Install MkDocs:

```bash
python -m pip install -r docs/requirements.txt
```

Build the site:

```bash
python -m mkdocs build
```

Serve locally while editing:

```bash
python -m mkdocs serve
```

## What to Validate Manually

Automated tests can check file contracts and command construction, but M7 still
requires visual judgment:

- anchor posterior quality after M4b;
- uncertainty histogram shape and outliers;
- map-scale colored anchor PLYs;
- held-out RGB/uncertainty overlays;
- NBV winner and top candidates.
