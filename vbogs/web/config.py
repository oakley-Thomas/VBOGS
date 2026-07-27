"""Validated GUI preset and request configuration."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigValidationError(ValueError):
    """Raised when a browser-supplied configuration is not safe or valid."""


FORBIDDEN_SEGMENTS = {
    "upload",
    "orchestration",
    "raw_root",
    "poses_root",
    "calibration_dir",
    "ncore_root",
    "model_path",
    "weights_path",
    "service_account_file",
    "folder_id",
    "rclone_args",
}


@dataclass(frozen=True)
class Preset:
    slug: str
    name: str
    description: str
    base_config: str
    datasets: tuple[str, ...]
    allowed_paths: frozenset[str]
    defaults: dict[str, Any]
    limits: dict[str, dict[str, float]]

    def public(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "name": self.name,
            "description": self.description,
            "base_config": self.base_config,
            "datasets": list(self.datasets),
            "allowed_paths": sorted(self.allowed_paths),
            "defaults": self.defaults,
            "limits": self.limits,
        }


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigValidationError(f"{label} must be a mapping")
    return value


def dotted_paths(value: dict[str, Any], prefix: str = "") -> set[str]:
    result: set[str] = set()
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(dotted_paths(item, path))
        else:
            result.add(path)
    return result


def set_dotted(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            raise ConfigValidationError(f"Cannot set nested configuration path {path}")
        cursor = child
    cursor[parts[-1]] = value


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ConfigValidationError(f"Could not read configuration: {path}") from exc
    return _mapping(raw, str(path))


def load_presets(root: Path, repo_root: Path) -> dict[str, Preset]:
    presets: dict[str, Preset] = {}
    for path in sorted(root.glob("*.yaml")):
        raw = load_yaml(path)
        slug = str(raw.get("slug") or path.stem)
        base_config = str(raw.get("base_config", ""))
        base_path = (repo_root / base_config).resolve()
        if not base_config or repo_root.resolve() not in base_path.parents or not base_path.is_file():
            raise ConfigValidationError(f"Preset {slug} has an invalid base_config")
        allowed = frozenset(str(item) for item in raw.get("allowed_paths", []))
        if not allowed:
            raise ConfigValidationError(f"Preset {slug} must define allowed_paths")
        presets[slug] = Preset(
            slug=slug,
            name=str(raw.get("name", slug)),
            description=str(raw.get("description", "")),
            base_config=base_config,
            datasets=tuple(str(item) for item in raw.get("datasets", ("kitti360",))),
            allowed_paths=allowed,
            defaults=_mapping(raw.get("defaults", {}), f"Preset {slug} defaults"),
            limits=_mapping(raw.get("limits", {}), f"Preset {slug} limits"),
        )
    return presets


def _validate_patch(preset: Preset, patch: dict[str, Any]) -> None:
    for path in dotted_paths(patch):
        segments = set(path.split("."))
        if segments & FORBIDDEN_SEGMENTS or path not in preset.allowed_paths:
            raise ConfigValidationError(f"Setting `{path}` is not allowed by preset {preset.slug}")
    for path, rule in preset.limits.items():
        if path not in dotted_paths(patch):
            continue
        cursor: Any = patch
        for part in path.split("."):
            cursor = cursor[part]
        if isinstance(cursor, bool) or not isinstance(cursor, (int, float)):
            continue
        if "minimum" in rule and cursor < rule["minimum"]:
            raise ConfigValidationError(f"`{path}` must be at least {rule['minimum']}")
        if "maximum" in rule and cursor > rule["maximum"]:
            raise ConfigValidationError(f"`{path}` must be at most {rule['maximum']}")


def resolve_config(
    preset: Preset,
    *,
    repo_root: Path,
    dataset: str,
    scene_id: str,
    overrides: dict[str, Any] | None,
    advanced_yaml: str | None,
) -> dict[str, Any]:
    if dataset not in preset.datasets:
        raise ConfigValidationError(f"Preset {preset.slug} does not support {dataset}")
    base = load_yaml((repo_root / preset.base_config).resolve())
    resolved = copy.deepcopy(base)
    patch = copy.deepcopy(preset.defaults)
    if overrides:
        if not isinstance(overrides, dict):
            raise ConfigValidationError("overrides must be a mapping")
        for path, value in overrides.items():
            if not isinstance(path, str):
                raise ConfigValidationError("override names must be strings")
            set_dotted(patch, path, value)
    if advanced_yaml:
        try:
            advanced = yaml.safe_load(advanced_yaml) or {}
        except yaml.YAMLError as exc:
            raise ConfigValidationError(f"Advanced settings are not valid YAML: {exc}") from exc
        advanced = _mapping(advanced, "advanced settings")
        patch.update(advanced)
    _validate_patch(preset, patch)
    for path in dotted_paths(patch):
        value: Any = patch
        for part in path.split("."):
            value = value[part]
        set_dotted(resolved, path, value)
    resolved.setdefault("dataset", {})["name"] = dataset
    resolved["dataset"]["scene_id"] = scene_id
    resolved.setdefault("pipeline", {})["drive"] = scene_id
    resolved.pop("upload", None)
    resolved.pop("orchestration", None)
    return resolved
