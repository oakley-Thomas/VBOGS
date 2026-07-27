from pathlib import Path

import pytest

from vbogs.web.config import ConfigValidationError, load_presets, resolve_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gui_preset_resolves_safe_override_and_removes_upload():
    preset = load_presets(REPO_ROOT / "configs/gui/presets", REPO_ROOT)["kitti360-dev"]
    resolved = resolve_config(
        preset,
        repo_root=REPO_ROOT,
        dataset="kitti360",
        scene_id="2013_05_28_drive_0008_sync",
        overrides={"train.iterations": 30000},
        advanced_yaml="render:\n  max_views: 4\n",
    )
    assert resolved["dataset"]["name"] == "kitti360"
    assert resolved["dataset"]["scene_id"] == "2013_05_28_drive_0008_sync"
    assert resolved["train"]["iterations"] == 30000
    assert resolved["render"]["max_views"] == 4
    assert "upload" not in resolved


def test_gui_preset_rejects_paths_and_unknown_advanced_keys():
    preset = load_presets(REPO_ROOT / "configs/gui/presets", REPO_ROOT)["kitti360-dev"]
    with pytest.raises(ConfigValidationError, match="not allowed"):
        resolve_config(
            preset, repo_root=REPO_ROOT, dataset="kitti360", scene_id="scene",
            overrides={"train.output_root": "/tmp/not-allowed"}, advanced_yaml=None,
        )
    with pytest.raises(ConfigValidationError, match="not allowed"):
        resolve_config(
            preset, repo_root=REPO_ROOT, dataset="kitti360", scene_id="scene",
            overrides=None, advanced_yaml="upload:\n  enabled: true\n",
        )
