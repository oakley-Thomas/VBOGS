import os

import pytest

from scripts.upload_google_drive import (
    build_parser,
    build_rclone_command,
    rclone_env,
)


def test_upload_defaults_to_curated_zip_from_config_defaults():
    parser = build_parser(
        {
            "drive": "drive_sync",
            "run_output_root": "outputs/v1_0",
        }
    )
    args = parser.parse_args(["--dry-run"])

    assert build_rclone_command(args) == [
        "rclone",
        "copyto",
        "outputs/v1_0/drive_sync.zip",
        "vbogs_gdrive:drive_sync.zip",
    ]


def test_upload_directory_uses_rclone_copy(tmp_path):
    source = tmp_path / "bundle"
    source.mkdir()
    parser = build_parser({"drive": "drive_sync"})
    args = parser.parse_args(
        [
            "--source",
            str(source),
            "--remote",
            "gdrive",
            "--dest",
            "runs/drive_sync",
            "--rclone-args",
            "--checksum --progress",
        ]
    )

    assert build_rclone_command(args) == [
        "rclone",
        "copy",
        str(source),
        "gdrive:runs/drive_sync",
        "--checksum",
        "--progress",
    ]


def test_service_account_env_config_is_derived_for_default_remote():
    parser = build_parser({"drive": "drive_sync"})
    args = parser.parse_args(
        [
            "--folder-id",
            "folder123",
            "--service-account-file",
            "/run/secrets/gdrive.json",
        ]
    )

    env = rclone_env(args, base_env={})

    assert env["RCLONE_CONFIG_VBOGS_GDRIVE_TYPE"] == "drive"
    assert env["RCLONE_CONFIG_VBOGS_GDRIVE_SCOPE"] == "drive"
    assert env["RCLONE_CONFIG_VBOGS_GDRIVE_ROOT_FOLDER_ID"] == "folder123"
    assert env["RCLONE_CONFIG_VBOGS_GDRIVE_SERVICE_ACCOUNT_FILE"] == "/run/secrets/gdrive.json"


def test_existing_custom_remote_does_not_force_env_config(monkeypatch):
    parser = build_parser({"drive": "drive_sync"})
    args = parser.parse_args(["--remote", "existing"])

    env = rclone_env(args, base_env={"PATH": os.environ.get("PATH", "")})

    assert "RCLONE_CONFIG_EXISTING_TYPE" not in env


def test_env_backed_remote_requires_env_safe_name():
    parser = build_parser({"drive": "drive_sync"})
    args = parser.parse_args(["--remote", "bad-name", "--folder-id", "folder123"])

    with pytest.raises(ValueError, match="letters, digits, and underscores"):
        rclone_env(args, base_env={})
