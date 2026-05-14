#!/usr/bin/env python3

"""Launch the VBOGS pipeline runner from compose environment variables."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
UPLOAD_ENV_OPTIONS = (
    ("VBOGS_GDRIVE_SOURCE", "--gdrive-source"),
    ("VBOGS_GDRIVE_REMOTE", "--gdrive-remote"),
    ("VBOGS_GDRIVE_DEST", "--gdrive-dest"),
    ("VBOGS_GDRIVE_FOLDER_ID", "--gdrive-folder-id"),
    ("VBOGS_GDRIVE_SERVICE_ACCOUNT_FILE", "--gdrive-service-account-file"),
    ("VBOGS_GDRIVE_SCOPE", "--gdrive-scope"),
    ("VBOGS_GDRIVE_RCLONE_ARGS", "--gdrive-rclone-args"),
)


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUE_VALUES


def append_env_option(cmd: list[str], env_name: str, flag: str) -> None:
    value = os.environ.get(env_name)
    if value:
        cmd.extend([flag, value])


def run_printed(cmd: list[str]) -> None:
    print("+ " + shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def upload_cmd(config: str, drive: str | None) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/upload_google_drive.py",
        "--config",
        config,
    ]
    if drive:
        cmd.extend(["--drive", drive])

    for env_name, flag in (
        ("VBOGS_GDRIVE_SOURCE", "--source"),
        ("VBOGS_GDRIVE_REMOTE", "--remote"),
        ("VBOGS_GDRIVE_DEST", "--dest"),
        ("VBOGS_GDRIVE_FOLDER_ID", "--folder-id"),
        ("VBOGS_GDRIVE_SERVICE_ACCOUNT_FILE", "--service-account-file"),
        ("VBOGS_GDRIVE_SCOPE", "--scope"),
        ("VBOGS_GDRIVE_RCLONE_ARGS", "--rclone-args"),
    ):
        append_env_option(cmd, env_name, flag)

    if truthy(os.environ.get("VBOGS_GDRIVE_DRY_RUN")):
        cmd.append("--dry-run")
    return cmd


def pipeline_cmd(config: str, drive: str | None) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_drive_pipeline.py",
        "--config",
        config,
        "--use-service-labels",
    ]

    if drive:
        cmd.extend(["--drive", drive])

    extra_args = os.environ.get("VBOGS_PIPELINE_ARGS", "")
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    if truthy(os.environ.get("VBOGS_GDRIVE_UPLOAD")):
        cmd.append("--upload-google-drive")
        for env_name, flag in UPLOAD_ENV_OPTIONS:
            append_env_option(cmd, env_name, flag)
        if truthy(os.environ.get("VBOGS_GDRIVE_DRY_RUN")):
            cmd.append("--gdrive-dry-run")

    return cmd


def main() -> None:
    config = os.environ.get("VBOGS_PIPELINE_CONFIG") or "pipeline_config.yaml"
    drive = os.environ.get("VBOGS_DRIVE")
    run_printed(pipeline_cmd(config, drive))


if __name__ == "__main__":
    main()
