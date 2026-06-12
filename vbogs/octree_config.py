"""Helpers for loading Octree-AnyGS configs from VBOGS entry points."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def resolve_relative_source_path(cfg: dict[str, Any], model_path: Path) -> dict[str, Any]:
    """Resolve relative `model_params.source_path` entries against `model_path`.

    Octree-AnyGS treats relative source paths as relative to the process working
    directory. VBOGS local-viewer exports instead store a portable
    `source_path: ../prepared` next to the copied model directory, so repo-owned
    loaders normalize that path before handing the config to upstream code.
    """

    resolved = copy.deepcopy(cfg)
    model_params = resolved.get("model_params")
    if not isinstance(model_params, dict):
        return resolved

    source_path = model_params.get("source_path")
    if not source_path:
        return resolved

    source = Path(str(source_path))
    if not source.is_absolute():
        model_params["source_path"] = str((model_path / source).resolve())
    return resolved


def load_octree_config_for_model(model_path: Path) -> dict[str, Any]:
    """Load `config.yaml` and normalize paths for repo-owned Octree loaders."""

    import yaml

    model_path = model_path.resolve()
    with (model_path / "config.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected YAML mapping in {model_path / 'config.yaml'}")
    return resolve_relative_source_path(cfg, model_path)
