#!/usr/bin/env python3

"""Upload curated VBOGS run artifacts to Google Drive with rclone."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_drive_pipeline import DEFAULT_CONFIG, config_path_arg, load_config_defaults


DEFAULT_REMOTE = "vbogs_gdrive"
DEFAULT_RUN_OUTPUT_ROOT = Path("outputs/v1_0")
TRUE_VALUES = {"1", "true", "yes", "y", "on"}
REMOTE_ENV_RE = re.compile(r"^[A-Za-z0-9_]+$")


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUE_VALUES


def build_parser(config_defaults: dict | None = None) -> argparse.ArgumentParser:
    defaults = config_defaults or {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=config_path_arg,
        default=DEFAULT_CONFIG,
        help=(
            "Pipeline config used to infer --drive and --run-output-root. "
            "Pass an empty string to disable config loading."
        ),
    )
    parser.add_argument(
        "--drive",
        default=defaults.get("drive"),
        help="KITTI-360 drive id. Defaults to `pipeline.drive` from --config.",
    )
    parser.add_argument(
        "--run-output-root",
        type=Path,
        default=defaults.get("run_output_root") or DEFAULT_RUN_OUTPUT_ROOT,
        help=(
            "Root containing curated pipeline outputs. The default source is "
            "`<run-output-root>/<drive>.zip`."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="File or directory to upload. Defaults to the curated run zip.",
    )
    parser.add_argument(
        "--remote",
        default=os.environ.get("VBOGS_GDRIVE_REMOTE", DEFAULT_REMOTE),
        help="rclone remote name. Defaults to VBOGS_GDRIVE_REMOTE or `vbogs_gdrive`.",
    )
    parser.add_argument(
        "--dest",
        default=os.environ.get("VBOGS_GDRIVE_DEST", ""),
        help="Destination path inside the remote/root folder.",
    )
    parser.add_argument(
        "--folder-id",
        default=os.environ.get("VBOGS_GDRIVE_FOLDER_ID", ""),
        help="Google Drive folder id used as rclone's root_folder_id.",
    )
    parser.add_argument(
        "--service-account-file",
        default=os.environ.get("VBOGS_GDRIVE_SERVICE_ACCOUNT_FILE", ""),
        help="Path to a Google service-account JSON file inside the container.",
    )
    parser.add_argument(
        "--service-account-credentials",
        default=os.environ.get("VBOGS_GDRIVE_SERVICE_ACCOUNT_CREDENTIALS", ""),
        help="Raw Google service-account JSON credentials.",
    )
    parser.add_argument(
        "--scope",
        default=os.environ.get("VBOGS_GDRIVE_SCOPE", ""),
        help="rclone Google Drive scope. Defaults to `drive` when creating env config.",
    )
    parser.add_argument(
        "--rclone-args",
        default=os.environ.get("VBOGS_GDRIVE_RCLONE_ARGS", ""),
        help="Extra arguments appended to the rclone command, parsed with shlex.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=truthy(os.environ.get("VBOGS_GDRIVE_DRY_RUN")),
        help="Print the rclone command without transferring files.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=config_path_arg, default=DEFAULT_CONFIG)
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_defaults = load_config_defaults(pre_args.config)
    parser = build_parser(config_defaults)
    args = parser.parse_args(argv)
    if not args.drive and args.source is None:
        parser.error("--drive is required when --source is not set")
    return args


def clean_remote_name(remote: str) -> str:
    remote_name = remote.rstrip(":").strip()
    if not remote_name:
        raise ValueError("Google Drive rclone remote name must not be empty")
    return remote_name


def remote_destination(remote: str, dest: str) -> str:
    remote_name = clean_remote_name(remote)
    dest_path = dest.strip("/")
    if dest_path:
        return f"{remote_name}:{dest_path}"
    return f"{remote_name}:"


def remote_file_destination(remote: str, dest: str, filename: str) -> str:
    remote_name = clean_remote_name(remote)
    dest_path = dest.strip("/")
    if dest_path:
        return f"{remote_name}:{dest_path}/{filename}"
    return f"{remote_name}:{filename}"


def default_source(drive: str, run_output_root: Path) -> Path:
    return Path(run_output_root) / f"{drive}.zip"


def source_kind(source: Path) -> str:
    if source.is_file():
        return "file"
    if source.is_dir():
        return "directory"
    if source.suffix:
        return "file"
    return "directory"


def rclone_env(args: argparse.Namespace, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    remote_name = clean_remote_name(args.remote)
    configure_from_env = (
        remote_name == DEFAULT_REMOTE
        or bool(args.folder_id)
        or bool(args.service_account_file)
        or bool(args.service_account_credentials)
    )
    if not configure_from_env:
        return env

    if not REMOTE_ENV_RE.match(remote_name):
        raise ValueError(
            "Environment-backed rclone remotes may only contain letters, digits, "
            f"and underscores: {remote_name!r}"
        )

    prefix = f"RCLONE_CONFIG_{remote_name.upper()}_"
    env.setdefault(f"{prefix}TYPE", "drive")
    env.setdefault(f"{prefix}SCOPE", args.scope or "drive")
    if args.folder_id:
        env.setdefault(f"{prefix}ROOT_FOLDER_ID", args.folder_id)
    if args.service_account_file:
        env.setdefault(f"{prefix}SERVICE_ACCOUNT_FILE", args.service_account_file)
    if args.service_account_credentials:
        env.setdefault(
            f"{prefix}SERVICE_ACCOUNT_CREDENTIALS",
            args.service_account_credentials,
        )
    return env


def build_rclone_command(args: argparse.Namespace) -> list[str]:
    source = args.source or default_source(args.drive, args.run_output_root)
    source = Path(source)
    if not args.dry_run and not source.exists():
        raise FileNotFoundError(f"Upload source does not exist: {source}")

    extra_args = shlex.split(args.rclone_args)
    if source_kind(source) == "file":
        destination = remote_file_destination(args.remote, args.dest, source.name)
        return ["rclone", "copyto", str(source), destination, *extra_args]

    destination = remote_destination(args.remote, args.dest)
    return ["rclone", "copy", str(source), destination, *extra_args]


def run_command(cmd: Sequence[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("+ " + shlex.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    env = rclone_env(args)
    cmd = build_rclone_command(args)
    run_command(cmd, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
