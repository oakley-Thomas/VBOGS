import re
import sys
from pathlib import Path

import pytest

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
    pattern = (
        rf"^  {re.escape(service)}:\n"
        r"(?P<block>.*?)(?=^  [\w-]+:|^[A-Za-z][\w-]*:|\Z)"
    )
    match = re.search(
        pattern,
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


def test_render_and_nbv_steps_forward_model_and_uncertainty_paths():
    parser = build_parser({})
    args = parser.parse_args(
        [
            "--drive",
            "drive_sync",
            "--model-path",
            "/data/OCTREE-ANYGS/drive_sync/run",
            "--bucket-iteration",
            "120000",
        ]
    )
    by_name = {step.name: step for step in build_steps(args)}

    render_step = by_name["render"]
    assert (
        render_step.command[render_step.command.index("--model-path") + 1]
        == "/data/OCTREE-ANYGS/drive_sync/run"
    )
    assert render_step.command[render_step.command.index("--uncertainty") + 1] == (
        "data/m4/drive_sync/U.npy"
    )
    assert render_step.command[render_step.command.index("--iteration") + 1] == "120000"

    nbv_step = by_name["nbv"]
    assert (
        nbv_step.command[nbv_step.command.index("--model-path") + 1]
        == "/data/OCTREE-ANYGS/drive_sync/run"
    )
    assert nbv_step.command[nbv_step.command.index("--u-path") + 1] == (
        "data/m4/drive_sync/U.npy"
    )
    assert nbv_step.command[nbv_step.command.index("--iteration") + 1] == "120000"


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


def test_kitti360_prepare_forwards_training_cameras():
    parser = build_parser({})
    args = parser.parse_args(
        [
            "--drive",
            "drive_sync",
            "--training-cameras",
            "stereo",
        ]
    )
    prepare_step = next(step for step in build_steps(args) if step.name == "prepare")

    assert prepare_step.command[:2] == ("python", "scripts/prepare_kitti360_colmap.py")
    training_index = prepare_step.command.index("--training-cameras")
    assert prepare_step.command[training_index + 1] == "stereo"


def test_config_default_sets_training_cameras(tmp_path):
    config_path = tmp_path / "pipeline_config.yaml"
    config_path.write_text(
        """
pipeline:
  drive: drive_sync
prepare:
  training_cameras: stereo
""",
        encoding="utf-8",
    )

    defaults = load_config_defaults(config_path)
    parser = build_parser(defaults)
    args = parser.parse_args([])
    prepare_step = next(step for step in build_steps(args) if step.name == "prepare")

    training_index = prepare_step.command.index("--training-cameras")
    assert prepare_step.command[training_index + 1] == "stereo"


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
            "configs/pipeline/portainer.yaml",
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
        "configs/pipeline/portainer.yaml",
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


def test_nvidia_ncore_pipeline_dispatches_prepare_and_points():
    parser = build_parser({})
    args = parser.parse_args(
        [
            "--dataset-name",
            "nvidia_ncore",
            "--scene-id",
            "clip_001",
            "--ncore-root",
            "/data/ncore",
            "--camera-id",
            "camera_front_wide_120fov,camera_front_tele_30fov",
            "--training-cameras",
            "stereo",
            "--point-source",
            "lidar",
        ]
    )
    by_name = {step.name: step for step in build_steps(args)}

    assert by_name["prepare"].command[:2] == (
        "python",
        "scripts/prepare_nvidia_ncore_colmap.py",
    )
    assert "--scene-id" in by_name["prepare"].command
    assert (
        by_name["prepare"].command[by_name["prepare"].command.index("--scene-id") + 1]
        == "clip_001"
    )
    assert "--ncore-root" in by_name["prepare"].command
    assert "--training-cameras" not in by_name["prepare"].command

    point_step = by_name["stereo"]
    assert point_step.command[:2] == ("python", "scripts/export_points_world.py")
    assert point_step.command[point_step.command.index("--dataset-name") + 1] == "nvidia_ncore"
    assert point_step.command[point_step.command.index("--scene-id") + 1] == "clip_001"
    assert point_step.command[point_step.command.index("--point-source") + 1] == "lidar"
    assert "data/m4/clip_001/U.npy" in by_name["render"].command


def test_environment_pipeline_configs_are_loadable():
    for config_name in (
        "configs/pipeline/dev.yaml",
        "configs/pipeline/portainer.yaml",
    ):
        defaults = load_config_defaults(REPO_ROOT / config_name)
        assert defaults["drive"] == "2013_05_28_drive_0007_sync"
        assert defaults["run_output_root"] == "outputs/v1_0"
        assert defaults["gaussian_type"] == "explicit3D"
        assert defaults["render_resolution"] == 2
        assert defaults["bucket_point_chunk_size"] == 1000000
        assert "train_port" not in defaults
    assert (
        load_config_defaults(REPO_ROOT / "configs/pipeline/dev.yaml")["bucket_max_points"]
        == 10000000
    )
    assert (
        load_config_defaults(REPO_ROOT / "configs/pipeline/portainer.yaml")[
            "bucket_max_points"
        ]
        == 0
    )


def test_dev_compose_binds_only_local_checkout_and_uses_dev_config():
    dev_compose = (REPO_ROOT / "docker/compose/dev.yml").read_text(encoding="utf-8")
    override_compose = (REPO_ROOT / "docker/compose/override.yml").read_text(
        encoding="utf-8"
    )
    dev_pipeline = service_block(dev_compose, "vbogs-pipeline")
    override_pipeline = service_block(override_compose, "vbogs-pipeline")
    dev_filebrowser = service_block(dev_compose, "vbogs-filebrowser")
    override_filebrowser = service_block(override_compose, "vbogs-filebrowser")

    assert "VBOGS_LOCAL_OUTPUTS" not in dev_compose
    assert "VBOGS_LOCAL_OUTPUTS" not in override_compose
    assert "source: vbogs-outputs" in dev_pipeline
    assert "source: vbogs-outputs" in override_pipeline
    assert "target: /srv/project" in dev_filebrowser
    assert "target: /srv/project" in override_filebrowser
    assert "target: /srv/outputs" not in dev_filebrowser
    assert "target: /srv/outputs" not in override_filebrowser
    assert "configs/pipeline/dev.yaml" in dev_compose
    assert "configs/pipeline/dev.yaml" in override_compose

    for compose_text in (dev_compose, override_compose):
        bind_blocks = re.findall(
            r"- type: bind\n\s+source: (?P<source>.+)\n\s+target: (?P<target>.+)",
            compose_text,
        )
        assert bind_blocks
        assert (".", "/workspace/VBOGS") in bind_blocks
        assert (
            "${DOCKER_HOST_SOCKET:-/var/run/docker.sock}",
            "/var/run/docker.sock",
        ) in bind_blocks


def test_host_compose_flags_are_not_pipeline_arguments():
    parser = build_parser({})

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--drive", "drive_sync", "--compose-file", "docker/compose/dev.yml"]
        )


def test_pipeline_image_includes_zip_tools():
    pipeline_dockerfile = (REPO_ROOT / "docker/pipeline.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "\n    ffmpeg \\" in pipeline_dockerfile
    assert "\n    zip \\" in pipeline_dockerfile
    assert "\n    unzip \\" in pipeline_dockerfile
    assert "\n    docker.io \\" in pipeline_dockerfile
    assert "scripts/bootstrap_stack_repo.py" in pipeline_dockerfile
    assert "vbogs-bootstrap-repo" in pipeline_dockerfile
    assert "docker:27-cli" not in pipeline_dockerfile
    assert "COPY --from=docker-cli" not in pipeline_dockerfile
    assert "openssh-server" not in pipeline_dockerfile
    assert "vbogs-transfer-sshd" not in pipeline_dockerfile


def test_gsplat_is_built_from_source_for_requested_cuda_arches():
    for dockerfile_name in ("docker/torch.Dockerfile", "docker/vbgs-render.Dockerfile"):
        dockerfile = (REPO_ROOT / dockerfile_name).read_text(encoding="utf-8")

        assert "gsplat==1.5.3" in dockerfile
        assert "--no-binary=gsplat" in dockerfile


def test_service_images_do_not_clone_vbogs_during_build():
    for dockerfile_name in (
        "docker/torch.Dockerfile",
        "docker/jax.Dockerfile",
        "docker/vbgs-render.Dockerfile",
        "docker/pipeline.Dockerfile",
    ):
        dockerfile = (REPO_ROOT / dockerfile_name).read_text(encoding="utf-8")

        assert "VBOGS_GIT_URL" not in dockerfile
        assert "git clone \"${VBOGS_GIT_URL}\"" not in dockerfile


def test_vbgs_render_image_can_host_realtime_viewer():
    dockerfile = (REPO_ROOT / "docker/vbgs-render.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "torch_scatter" in dockerfile
    assert "gsplat==1.5.3" in dockerfile
    assert "kornia==0.7.4" in dockerfile
    assert "fastapi==0.115.14" in dockerfile
    assert "uvicorn[standard]==0.34.3" in dockerfile
    assert "/workspace/VBOGS/Octree-AnyGS" in dockerfile
    assert "EXPOSE 8070" in dockerfile


def test_pipeline_compose_mounts_match_shared_stack_volumes():
    shared_targets = [
        "/workspace/VBOGS",
        "/workspace/VBOGS/data",
        "/workspace/VBOGS/data/KITTI-360",
        "/workspace/VBOGS/data/NVIDIA-PhysicalAI-AV-NCore",
        "/workspace/VBOGS/outputs",
        "/workspace/VBOGS/generated_configs",
        "/data/COLMAP",
        "/data/OCTREE-ANYGS",
    ]

    for compose_name in (
        "docker/compose/compose.yml",
        "docker/compose/deploy.yml",
        "docker/compose/portainer-build.yml",
        "docker/compose/portainer-local.yml",
    ):
        pipeline = service_block(
            (REPO_ROOT / compose_name).read_text(encoding="utf-8"),
            "vbogs-pipeline",
        )
        for target in shared_targets:
            assert f"target: {target}" in pipeline


def test_stack_compose_files_only_bind_docker_socket_for_pipeline():
    for compose_name in (
        "docker/compose/compose.yml",
        "docker/compose/deploy.yml",
        "docker/compose/portainer-build.yml",
        "docker/compose/portainer-local.yml",
    ):
        compose_text = (REPO_ROOT / compose_name).read_text(encoding="utf-8")

        assert "source: ${DOCKER_HOST_SOCKET:-/var/run/docker.sock}" in compose_text
        assert "target: /var/run/docker.sock" in compose_text
        for service in (
            "vbogs-torch",
            "vbogs-jax",
            "vbogs-vbgs-render",
            "vbogs-filebrowser",
        ):
            assert "type: bind" not in service_block(compose_text, service)


def test_vbgs_render_service_publishes_viewer_port():
    expected_pythonpath = (
        "PYTHONPATH: /workspace/VBOGS:/workspace/VBOGS/Octree-AnyGS:"
        "/workspace/VBOGS/vbgs:/workspace/gaussian-splatting"
    )

    for compose_name in (
        "docker/compose/compose.yml",
        "docker/compose/deploy.yml",
        "docker/compose/portainer-build.yml",
        "docker/compose/portainer-local.yml",
    ):
        compose_text = (REPO_ROOT / compose_name).read_text(encoding="utf-8")
        render = service_block(compose_text, "vbogs-vbgs-render")

        assert "VBOGS_RENDER_VIEWER_HOST_BIND:-0.0.0.0" in render
        assert "VBOGS_RENDER_VIEWER_HOST_PORT:-8071" in render
        assert ":8070" in render
        assert expected_pythonpath in render
        assert "*vbogs-octree-anygs-mount" in render or "target: /data/OCTREE-ANYGS" in render


def test_compose_uses_filebrowser_instead_of_transfer_sidecar():
    filebrowser_targets = [
        "/srv/project",
        "/srv/data",
        "/srv/data/KITTI-360",
        "/srv/data/NVIDIA-PhysicalAI-AV-NCore",
        "/srv/outputs",
        "/srv/generated_configs",
        "/srv/COLMAP",
        "/srv/OCTREE-ANYGS",
    ]

    for compose_name in (
        "docker/compose/compose.yml",
        "docker/compose/deploy.yml",
        "docker/compose/portainer-build.yml",
        "docker/compose/portainer-local.yml",
    ):
        compose_text = (REPO_ROOT / compose_name).read_text(encoding="utf-8")
        filebrowser = service_block(compose_text, "vbogs-filebrowser")

        assert "vbogs-transfer:" not in compose_text
        assert "filebrowser/filebrowser:v2-s6" in filebrowser
        assert "VBOGS_FILEBROWSER_HOST_PORT:-8088" in filebrowser
        assert "FB_DISABLE_EXEC" in filebrowser
        assert "vbogs-filebrowser-database" in filebrowser
        assert "vbogs-filebrowser-config" in filebrowser
        for target in filebrowser_targets:
            assert f"target: {target}" in filebrowser
        assert filebrowser.count("read_only: true") >= len(filebrowser_targets)


def test_portainer_compose_uses_portainer_config():
    portainer_compose = (REPO_ROOT / "docker/compose/deploy.yml").read_text(
        encoding="utf-8"
    )
    portainer_build_compose = (
        REPO_ROOT / "docker/compose/portainer-build.yml"
    ).read_text(encoding="utf-8")
    portainer_local_compose = (
        REPO_ROOT / "docker/compose/portainer-local.yml"
    ).read_text(encoding="utf-8")
    stack_env = (REPO_ROOT / "configs/docker/stack.env").read_text(encoding="utf-8")

    assert "configs/pipeline/portainer.yaml" in portainer_compose
    assert "configs/pipeline/portainer.yaml" in portainer_build_compose
    assert "configs/pipeline/portainer.yaml" in portainer_local_compose
    assert "VBOGS_PIPELINE_CONFIG=configs/pipeline/portainer.yaml" in stack_env
    assert "NVIDIA_DRIVER_CAPABILITIES: compute,utility" in portainer_compose
    assert "NVIDIA_DRIVER_CAPABILITIES: compute,utility" in portainer_build_compose
    assert "NVIDIA_DRIVER_CAPABILITIES: compute,utility" in portainer_local_compose
    assert "VBOGS_GDRIVE_UPLOAD" in portainer_compose
    assert "VBOGS_GDRIVE_UPLOAD" in portainer_build_compose
    assert "VBOGS_GDRIVE_UPLOAD" in portainer_local_compose
    assert "target: /workspace/VBOGS/outputs" in portainer_compose
    assert "target: /workspace/VBOGS/outputs" in portainer_build_compose
    assert "target: /workspace/VBOGS/outputs" in portainer_local_compose
    assert "VBOGS_FILEBROWSER_IMAGE=filebrowser/filebrowser:v2-s6" in stack_env
    assert "VBOGS_FILEBROWSER_HOST_PORT=8088" in stack_env
    assert "VBOGS_RENDER_VIEWER_HOST_BIND=0.0.0.0" in stack_env
    assert "VBOGS_RENDER_VIEWER_HOST_PORT=8071" in stack_env
    assert "VBOGS_TRANSFER_AUTHORIZED_KEYS" not in stack_env
    assert "VBOGS_PIPELINE_AUTORUN" not in stack_env
    assert "VBOGS_PIPELINE_ARGS" not in stack_env
    assert "VBOGS_PIPELINE_AUTORUN" not in portainer_compose
    assert "VBOGS_PIPELINE_AUTORUN" not in portainer_build_compose
    assert "VBOGS_PIPELINE_AUTORUN" not in portainer_local_compose


def test_portainer_build_compose_builds_local_images():
    portainer_build_compose = (
        REPO_ROOT / "docker/compose/portainer-build.yml"
    ).read_text(encoding="utf-8")

    assert "pull_policy: build" in portainer_build_compose
    assert "build: *vbogs-torch-build" in portainer_build_compose
    assert "build: *vbogs-jax-build" in portainer_build_compose
    assert "build: *vbogs-vbgs-render-build" in portainer_build_compose
    assert "build: *vbogs-pipeline-build" in portainer_build_compose
    assert "oakleyth/vbogs" not in portainer_build_compose
    assert "VBOGS_GIT_URL" not in portainer_build_compose


def test_portainer_local_compose_uses_cached_local_images():
    portainer_local_compose = (
        REPO_ROOT / "docker/compose/portainer-local.yml"
    ).read_text(encoding="utf-8")

    assert "build:" not in portainer_local_compose
    assert "pull_policy: never" in portainer_local_compose
    assert "local/vbogs-torch" in portainer_local_compose
    assert "local/vbogs-jax" in portainer_local_compose
    assert "local/vbogs-vbgs-render" in portainer_local_compose
    assert "local/vbogs-pipeline" in portainer_local_compose
    assert "oakleyth/vbogs" not in portainer_local_compose
