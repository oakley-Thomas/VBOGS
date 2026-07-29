import argparse
import subprocess

import pytest

from scripts.train_octree_anygs import build_config, relay_training_process, resolve_gui_port


def make_args(tmp_path, *, gaussian_type="implicit3D"):
    return argparse.Namespace(
        dataset_path=tmp_path / "COLMAP" / "drive_sync",
        scene_name="",
        dataset_name="kitti360",
        output_root=tmp_path / "OCTREE-ANYGS",
        resolution=4,
        iterations=30000,
        llffhold=8,
        eval=True,
        gaussian_type=gaussian_type,
        feat_dim=12,
        base_layer=8,
        visible_threshold=0.03,
        port=None,
    )


def test_build_config_keeps_implicit_neural_defaults(tmp_path):
    cfg = build_config(make_args(tmp_path))

    model_kwargs = cfg["model_params"]["model_config"]["kwargs"]
    optim_params = cfg["optim_params"]

    assert model_kwargs["gs_attr"] == "implicit3D"
    assert model_kwargs["color_attr"] == "RGB"
    assert model_kwargs["feat_dim"] == 12
    assert model_kwargs["base_layer"] == 8
    assert model_kwargs["visible_threshold"] == 0.03
    assert optim_params["mlp_opacity_lr_max_steps"] == 30000
    assert "opacity_lr" not in optim_params
    assert cfg["model_params"]["eval"] is True


def test_build_config_can_select_explicit_3d_gaussians(tmp_path):
    cfg = build_config(make_args(tmp_path, gaussian_type="explicit3D"))

    model_kwargs = cfg["model_params"]["model_config"]["kwargs"]
    optim_params = cfg["optim_params"]

    assert model_kwargs["gs_attr"] == "explicit3D"
    assert model_kwargs["color_attr"] == "SH2"
    assert model_kwargs["render_mode"] == "RGB"
    assert model_kwargs["base_layer"] == 8
    assert model_kwargs["visible_threshold"] == 0.03
    assert "feat_dim" not in model_kwargs
    assert "n_offsets" not in model_kwargs

    assert optim_params["feature_lr"] == 0.0025
    assert optim_params["opacity_lr"] == 0.05
    assert optim_params["scaling_lr"] == 0.005
    assert optim_params["rotation_lr"] == 0.001
    assert optim_params["lambda_dreg"] == 0.0
    assert "mlp_opacity_lr_init" not in optim_params
    assert "mlp_color_lr_init" not in optim_params


def test_resolve_gui_port_offsets_by_gpu_index():
    assert resolve_gui_port("0", None) == 6009
    assert resolve_gui_port("1", None) == 6010
    assert resolve_gui_port("-1", None) == 6009
    assert resolve_gui_port("cuda:1", None) == 6009
    assert resolve_gui_port("1", 6200) == 6200


def test_build_config_can_disable_octree_eval_split(tmp_path):
    args = make_args(tmp_path)
    args.eval = False

    cfg = build_config(args)

    assert cfg["model_params"]["eval"] is False


def test_build_config_enables_native_alpha_masks_when_requested(tmp_path):
    args = make_args(tmp_path)
    args.use_masks = True
    assert build_config(args)["model_params"]["add_mask"] is True


def test_training_relay_extracts_iteration_progress_and_reports_finalization(tmp_path, monkeypatch):
    calls = []

    class FakeStdout:
        def __init__(self):
            self.chunks = [
                b"\rTraining progress:  10%|##| 10/100 [00:01<00:09]",
                b"\rTraining progress: 100%|##| 100/100 [00:10<00:00]",
            ]

        def read1(self, _size):
            return self.chunks.pop(0) if self.chunks else b""

        def close(self):
            pass

    class FakeProcess:
        stdout = FakeStdout()

        def wait(self):
            return 0

    monkeypatch.setattr("scripts.train_octree_anygs.subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr("scripts.train_octree_anygs.write_progress", lambda _path, **payload: calls.append(payload))

    relay_training_process(["upstream-train"], cwd=str(tmp_path), progress_path=tmp_path / "progress.json", total_iterations=100)

    assert calls == [
        {"state": "starting", "current_iterations": 0, "total_iterations": 100},
        {"state": "running", "current_iterations": 10, "total_iterations": 100},
        {"state": "finalizing", "current_iterations": 100, "total_iterations": 100},
        {"state": "completed", "current_iterations": 100, "total_iterations": 100},
    ]


def test_training_relay_reports_failure(tmp_path, monkeypatch):
    calls = []

    class FakeStdout:
        def __init__(self):
            self.chunks = [b"\rTraining progress:  20%|##| 20/100"]

        def read1(self, _size):
            return self.chunks.pop(0) if self.chunks else b""

        def close(self):
            pass

    class FakeProcess:
        stdout = FakeStdout()

        def __init__(self):
            self.sent = False

        def wait(self):
            return 1

    process = FakeProcess()
    monkeypatch.setattr("scripts.train_octree_anygs.subprocess.Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr("scripts.train_octree_anygs.write_progress", lambda _path, **payload: calls.append(payload))

    with pytest.raises(subprocess.CalledProcessError):
        relay_training_process(["upstream-train"], cwd=str(tmp_path), progress_path=tmp_path / "progress.json", total_iterations=100)

    assert calls[-1] == {"state": "failed", "current_iterations": 20, "total_iterations": 100}
