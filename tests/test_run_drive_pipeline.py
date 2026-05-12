from pathlib import Path

from scripts.run_drive_pipeline import (
    TORCH_SERVICE,
    build_parser,
    build_steps,
    load_config_defaults,
    selected_steps,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_output_root_routes_v1_stage_outputs():
    parser = build_parser({})
    args = parser.parse_args(
        [
            "--drive",
            "2013_05_28_drive_0007_sync",
            "--run-output-root",
            "outputs/v1_0",
            "--start-at",
            "map-viz",
            "--stop-after",
            "bundle",
        ]
    )
    steps = selected_steps(build_steps(args), args.start_at, args.stop_after)
    by_name = {step.name: step for step in steps}

    assert list(by_name) == ["map-viz", "render", "nbv", "nbv-viz", "bundle"]
    assert by_name["map-viz"].service == TORCH_SERVICE
    assert "outputs/v1_0/2013_05_28_drive_0007_sync/pointclouds/anchors" in by_name["map-viz"].command
    assert "outputs/v1_0/2013_05_28_drive_0007_sync/views" in by_name["render"].command
    render_resolution_index = by_name["render"].command.index("--resolution")
    assert by_name["render"].command[render_resolution_index + 1] == "2"
    assert "outputs/v1_0/2013_05_28_drive_0007_sync/nbv" in by_name["nbv"].command
    assert "outputs/v1_0/2013_05_28_drive_0007_sync/nbv/viz" in by_name["nbv-viz"].command
    assert "outputs/v1_0/2013_05_28_drive_0007_sync" in by_name["bundle"].command


def test_run_output_root_keeps_explicit_stage_output_override():
    parser = build_parser({})
    args = parser.parse_args(
        [
            "--drive",
            "drive_sync",
            "--run-output-root",
            "outputs/v1_0",
            "--render-output-dir",
            "outputs/custom_views",
        ]
    )
    render_step = next(step for step in build_steps(args) if step.name == "render")

    assert "outputs/custom_views" in render_step.command
    assert "outputs/v1_0/drive_sync/views" not in render_step.command


def test_render_step_forwards_resolution_override():
    parser = build_parser({})
    args = parser.parse_args(
        [
            "--drive",
            "drive_sync",
            "--render-resolution",
            "1",
        ]
    )
    render_step = next(step for step in build_steps(args) if step.name == "render")

    resolution_index = render_step.command.index("--resolution")
    assert render_step.command[resolution_index + 1] == "1"


def test_train_step_forwards_gaussian_type():
    parser = build_parser({})
    args = parser.parse_args(
        [
            "--drive",
            "drive_sync",
            "--gaussian-type",
            "explicit3D",
        ]
    )
    train_step = next(step for step in build_steps(args) if step.name == "train")

    gaussian_type_index = train_step.command.index("--gaussian-type")
    assert train_step.command[gaussian_type_index + 1] == "explicit3D"


def test_config_default_sets_gaussian_type():
    parser = build_parser({"gaussian_type": "explicit3D"})
    args = parser.parse_args(["--drive", "drive_sync"])
    train_step = next(step for step in build_steps(args) if step.name == "train")

    gaussian_type_index = train_step.command.index("--gaussian-type")
    assert train_step.command[gaussian_type_index + 1] == "explicit3D"


def test_bucket_step_forwards_point_controls():
    parser = build_parser({})
    args = parser.parse_args(
        [
            "--drive",
            "drive_sync",
            "--bucket-point-chunk-size",
            "250000",
            "--bucket-max-points",
            "5000000",
        ]
    )
    bucket_step = next(step for step in build_steps(args) if step.name == "bucket")

    chunk_index = bucket_step.command.index("--point-chunk-size")
    max_points_index = bucket_step.command.index("--max-points")
    assert bucket_step.command[chunk_index + 1] == "250000"
    assert bucket_step.command[max_points_index + 1] == "5000000"


def test_environment_pipeline_configs_are_loadable():
    for config_name in ("pipeline_config.dev.yaml", "pipeline_config.portainer.yaml"):
        defaults = load_config_defaults(REPO_ROOT / config_name)
        assert defaults["drive"] == "2013_05_28_drive_0007_sync"
        assert defaults["run_output_root"] == "outputs/v1_0"
        assert defaults["gaussian_type"] == "explicit3D"
        assert defaults["render_resolution"] == 2
        assert defaults["bucket_point_chunk_size"] == 1000000
    assert load_config_defaults(REPO_ROOT / "pipeline_config.dev.yaml")["bucket_max_points"] == 10000000
    assert load_config_defaults(REPO_ROOT / "pipeline_config.portainer.yaml")["bucket_max_points"] == 0


def test_dev_compose_binds_local_outputs_and_uses_dev_config():
    dev_compose = (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")
    override_compose = (REPO_ROOT / "docker-compose.override.yml").read_text(encoding="utf-8")

    assert "${VBOGS_LOCAL_OUTPUTS:-./outputs}" in dev_compose
    assert "${VBOGS_LOCAL_OUTPUTS:-./outputs}" in override_compose
    assert "pipeline_config.dev.yaml" in dev_compose
    assert "pipeline_config.dev.yaml" in override_compose


def test_portainer_compose_uses_portainer_config():
    portainer_compose = (REPO_ROOT / "docker-compose.portainer.yml").read_text(encoding="utf-8")
    stack_env = (REPO_ROOT / "stack.env").read_text(encoding="utf-8")

    assert "pipeline_config.portainer.yaml" in portainer_compose
    assert "VBOGS_PIPELINE_CONFIG=pipeline_config.portainer.yaml" in stack_env
