import re
import sys
from pathlib import Path

from scripts.run_drive_pipeline import (
    TORCH_SERVICE,
    build_parser,
    build_upload_command,
    build_steps,
    load_config_defaults,
    selected_steps,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def service_block(compose_text: str, service: str) -> str:
    match = re.search(
        rf"^  {re.escape(service)}:\n(?P<block>.*?)(?=^  [\w-]+:|\Z)",
        compose_text,
        re.M | re.S,
    )
    assert match is not None
    return match.group("block")


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


def test_train_step_forwards_explicit_port_override():
    parser = build_parser({})
    args = parser.parse_args(
        [
            "--drive",
            "drive_sync",
            "--train-port",
            "6010",
        ]
    )
    train_step = next(step for step in build_steps(args) if step.name == "train")

    port_index = train_step.command.index("--port")
    assert train_step.command[port_index + 1] == "6010"


def test_config_default_sets_gaussian_type():
    parser = build_parser({"gaussian_type": "explicit3D"})
    args = parser.parse_args(["--drive", "drive_sync"])
    train_step = next(step for step in build_steps(args) if step.name == "train")

    gaussian_type_index = train_step.command.index("--gaussian-type")
    assert train_step.command[gaussian_type_index + 1] == "explicit3D"


def test_config_default_enables_google_drive_upload(tmp_path):
    config_path = tmp_path / "pipeline_config.yaml"
    config_path.write_text(
        """
pipeline:
  drive: drive_sync
outputs:
  run_root: outputs/custom
upload:
  enabled: true
  dest: experiments
  folder_id: folder123
  service_account_file: /run/secrets/gdrive.json
  dry_run: true
""",
        encoding="utf-8",
    )

    defaults = load_config_defaults(config_path)
    parser = build_parser(defaults)
    args = parser.parse_args([])

    assert args.upload_google_drive is True
    assert args.gdrive_dest == "experiments"
    assert args.gdrive_folder_id == "folder123"
    assert args.gdrive_service_account_file == "/run/secrets/gdrive.json"


def test_google_drive_upload_command_uses_curated_zip_defaults(monkeypatch):
    monkeypatch.setenv(
        "VBOGS_GDRIVE_SERVICE_ACCOUNT_CREDENTIALS",
        '{"private_key":"secret"}',
    )
    parser = build_parser(
        {
            "drive": "drive_sync",
            "run_output_root": "outputs/custom",
        }
    )
    args = parser.parse_args(
        [
            "--config",
            "pipeline_config.portainer.yaml",
            "--upload-google-drive",
            "--gdrive-folder-id",
            "folder123",
            "--gdrive-service-account-file",
            "/run/secrets/gdrive.json",
            "--gdrive-dest",
            "runs",
            "--gdrive-dry-run",
        ]
    )

    cmd = build_upload_command(args)

    assert cmd[:4] == [
        sys.executable,
        "scripts/upload_google_drive.py",
        "--config",
        "pipeline_config.portainer.yaml",
    ]
    assert "--drive" in cmd
    assert cmd[cmd.index("--drive") + 1] == "drive_sync"
    assert "--run-output-root" in cmd
    assert cmd[cmd.index("--run-output-root") + 1] == "outputs/custom"
    assert "--folder-id" in cmd
    assert "--service-account-file" in cmd
    assert "--dry-run" in cmd
    assert "--service-account-credentials" not in cmd
    assert '{"private_key":"secret"}' not in cmd


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
        assert "train_port" not in defaults
    assert load_config_defaults(REPO_ROOT / "pipeline_config.dev.yaml")["bucket_max_points"] == 10000000
    assert load_config_defaults(REPO_ROOT / "pipeline_config.portainer.yaml")["bucket_max_points"] == 0


def test_dev_compose_binds_local_outputs_and_uses_dev_config():
    dev_compose = (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")
    override_compose = (REPO_ROOT / "docker-compose.override.yml").read_text(encoding="utf-8")
    dev_pipeline = service_block(dev_compose, "vbogs-pipeline")
    override_pipeline = service_block(override_compose, "vbogs-pipeline")

    assert "${VBOGS_LOCAL_OUTPUTS:-./outputs}" in dev_compose
    assert "${VBOGS_LOCAL_OUTPUTS:-./outputs}" in override_compose
    assert "${VBOGS_LOCAL_OUTPUTS:-./outputs}" in dev_pipeline
    assert "${VBOGS_LOCAL_OUTPUTS:-./outputs}" in override_pipeline
    assert "pipeline_config.dev.yaml" in dev_compose
    assert "pipeline_config.dev.yaml" in override_compose


def test_pipeline_image_includes_zip_tools():
    pipeline_dockerfile = (REPO_ROOT / "docker/pipeline.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "\n    zip \\" in pipeline_dockerfile
    assert "\n    unzip \\" in pipeline_dockerfile


def test_pipeline_compose_mounts_match_shared_stack_volumes():
    shared_targets = [
        "/workspace/VBOGS/data",
        "/workspace/VBOGS/data/KITTI-360",
        "/workspace/VBOGS/outputs",
        "/workspace/VBOGS/generated_configs",
        "/data/COLMAP",
        "/data/OCTREE-ANYGS",
    ]

    for compose_name in ("docker-compose.yml", "docker-compose.portainer.yml"):
        pipeline = service_block(
            (REPO_ROOT / compose_name).read_text(encoding="utf-8"),
            "vbogs-pipeline",
        )
        for target in shared_targets:
            assert f"target: {target}" in pipeline


def test_portainer_compose_uses_portainer_config():
    portainer_compose = (REPO_ROOT / "docker-compose.portainer.yml").read_text(encoding="utf-8")
    stack_env = (REPO_ROOT / "stack.env").read_text(encoding="utf-8")

    assert "pipeline_config.portainer.yaml" in portainer_compose
    assert "VBOGS_PIPELINE_CONFIG=pipeline_config.portainer.yaml" in stack_env
    assert "NVIDIA_DRIVER_CAPABILITIES: compute,utility" in portainer_compose
    assert "VBOGS_GDRIVE_UPLOAD" in portainer_compose
    assert "target: /workspace/VBOGS/outputs" in portainer_compose
