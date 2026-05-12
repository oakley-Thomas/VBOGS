import argparse

from scripts.train_octree_anygs import build_config


def make_args(tmp_path, *, gaussian_type="implicit3D"):
    return argparse.Namespace(
        dataset_path=tmp_path / "COLMAP" / "drive_sync",
        scene_name="",
        dataset_name="kitti360",
        output_root=tmp_path / "OCTREE-ANYGS",
        resolution=4,
        iterations=30000,
        llffhold=8,
        gaussian_type=gaussian_type,
        feat_dim=12,
        base_layer=8,
        visible_threshold=0.03,
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
