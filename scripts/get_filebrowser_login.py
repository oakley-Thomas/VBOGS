#!/usr/bin/env python3
"""Print the File Browser admin login from the sibling service logs."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence


DEFAULT_SERVICE = "vbogs-filebrowser"
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
COMPOSE_SERVICE_LABEL = "com.docker.compose.service"

PASSWORD_PATTERNS = (
    re.compile(
        r"Randomly generated password for user\s+['\"]?"
        r"(?P<username>[^'\":\s]+)['\"]?\s*:\s*(?P<password>\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"Generated random password for user\s+['\"]?"
        r"(?P<username>[^'\":\s]+)['\"]?\s*:\s*(?P<password>\S+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"Generated random admin password\s*:?\s*(?P<password>\S+)",
        re.IGNORECASE,
    ),
)

ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(frozen=True)
class FilebrowserLogin:
    username: str
    password: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the vbogs-filebrowser container from inside "
            "vbogs-pipeline and print the generated admin login from "
            "Docker logs."
        )
    )
    parser.add_argument(
        "--container",
        default="",
        help="Explicit File Browser container name or id. Skips label lookup.",
    )
    parser.add_argument(
        "--service",
        default=DEFAULT_SERVICE,
        help=f"Compose service label to resolve. Default: {DEFAULT_SERVICE}",
    )
    parser.add_argument(
        "--label-project",
        default=os.environ.get("VBOGS_COMPOSE_PROJECT", ""),
        help=(
            "Compose project or Portainer stack label. Defaults to "
            "VBOGS_COMPOSE_PROJECT or the current container's compose label."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "shell"),
        default="text",
        help="Output format. Default: text.",
    )
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help=(
            "Generate a new password and set it on the File Browser user. "
            "This briefly stops the File Browser container to avoid a database "
            "lock. Use it when the one-time generated password is no longer "
            "in logs."
        ),
    )
    parser.add_argument(
        "--username",
        default="admin",
        help="File Browser username to reset with --reset-password. Default: admin.",
    )
    parser.add_argument(
        "--password",
        default="",
        help=(
            "Explicit password to set with --reset-password. If omitted, "
            "a random password is generated."
        ),
    )
    parser.add_argument(
        "--database",
        default="/database/filebrowser.db",
        help="File Browser database path inside the filebrowser container.",
    )
    parser.add_argument(
        "--config-file",
        default="/config/settings.json",
        help="File Browser config path inside the filebrowser container.",
    )
    return parser


def docker_output(args: Sequence[str], *, combine_stderr: bool = False) -> str:
    stderr = subprocess.STDOUT if combine_stderr else subprocess.PIPE
    try:
        completed = subprocess.run(
            ["docker", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "docker was not found. Run this from vbogs-pipeline after "
            "rebuilding the pipeline image."
        ) from exc
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or "").strip()
        if not combine_stderr:
            output = (exc.stderr or "").strip() or output
        detail = f": {output}" if output else ""
        raise RuntimeError(f"docker {' '.join(args)} failed{detail}") from exc
    return completed.stdout


def docker_json(args: Sequence[str]) -> object:
    return json.loads(docker_output(args))


def current_container_project() -> str:
    hostname = os.environ.get("HOSTNAME", "")
    if not hostname:
        return ""
    try:
        project = docker_output(
            [
                "inspect",
                "-f",
                f"{{{{ index .Config.Labels \"{COMPOSE_PROJECT_LABEL}\" }}}}",
                hostname,
            ]
        ).strip()
    except RuntimeError:
        return ""
    if project == "<no value>":
        return ""
    return project


def ps_matches(filters: Sequence[str], *, all_containers: bool) -> list[str]:
    args = ["ps", "-aq" if all_containers else "-q"]
    for docker_filter in filters:
        args.extend(["--filter", docker_filter])
    output = docker_output(args)
    return [line.strip() for line in output.splitlines() if line.strip()]


def resolve_service_container(service: str, *, project: str = "") -> str:
    filters = [f"label={COMPOSE_SERVICE_LABEL}={service}"]
    if project:
        filters.append(f"label={COMPOSE_PROJECT_LABEL}={project}")

    running_matches = ps_matches(filters, all_containers=False)
    if len(running_matches) == 1:
        return running_matches[0]
    if len(running_matches) > 1:
        raise RuntimeError(
            f"found {len(running_matches)} running containers for service "
            f"{service}; pass --label-project or --container"
        )

    all_matches = ps_matches(filters, all_containers=True)
    if len(all_matches) == 1:
        return all_matches[0]

    project_hint = f" in project {project}" if project else ""
    raise RuntimeError(
        f"expected exactly one container for service {service}{project_hint}; "
        f"found {len(all_matches)}"
    )


def strip_ansi(text: str) -> str:
    return ANSI_PATTERN.sub("", text)


def find_filebrowser_login(log_text: str) -> FilebrowserLogin | None:
    login: FilebrowserLogin | None = None
    for raw_line in log_text.splitlines():
        line = strip_ansi(raw_line)
        for pattern in PASSWORD_PATTERNS:
            match = pattern.search(line)
            if match is None:
                continue
            username = match.groupdict().get("username") or "admin"
            login = FilebrowserLogin(
                username=username,
                password=match.group("password"),
            )
    return login


def container_display_name(container: str) -> str:
    try:
        name = docker_output(["inspect", "-f", "{{ .Name }}", container]).strip()
    except RuntimeError:
        return container
    return name.removeprefix("/") or container


def published_ports(container: str) -> list[str]:
    try:
        output = docker_output(["port", container, "80/tcp"])
    except RuntimeError:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def generate_password() -> str:
    return secrets.token_urlsafe(24)


def inspect_container(container: str) -> dict:
    data = docker_json(["inspect", container])
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"docker inspect returned no data for {container}")
    inspected = data[0]
    if not isinstance(inspected, dict):
        raise RuntimeError(f"docker inspect returned invalid data for {container}")
    return inspected


def mount_args_for_cli(inspected: dict) -> list[str]:
    required = {"/config", "/database"}
    mounts = inspected.get("Mounts", [])
    if not isinstance(mounts, list):
        raise RuntimeError("docker inspect returned invalid mount data")

    by_destination = {}
    for mount in mounts:
        if isinstance(mount, dict) and mount.get("Destination") in required:
            by_destination[mount["Destination"]] = mount

    missing = sorted(required - set(by_destination))
    if missing:
        raise RuntimeError(
            "File Browser container is missing required mounts: "
            + ", ".join(missing)
        )

    args: list[str] = []
    for destination in sorted(required):
        mount = by_destination[destination]
        mount_type = mount.get("Type")
        if mount_type == "volume":
            source = mount.get("Name")
        elif mount_type == "bind":
            source = mount.get("Source")
        else:
            raise RuntimeError(
                f"unsupported mount type for {destination}: {mount_type}"
            )
        if not source:
            raise RuntimeError(f"could not resolve mount source for {destination}")
        args.extend(
            [
                "--mount",
                f"type={mount_type},source={source},target={destination}",
            ]
        )
    return args


def reset_filebrowser_password(
    container: str,
    *,
    username: str,
    password: str,
    database: str,
    config_file: str,
) -> FilebrowserLogin:
    inspected = inspect_container(container)
    image = inspected.get("Config", {}).get("Image")
    if not image:
        raise RuntimeError(f"could not resolve image for {container}")
    was_running = bool(inspected.get("State", {}).get("Running"))
    mount_args = mount_args_for_cli(inspected)

    try:
        if was_running:
            docker_output(["stop", container], combine_stderr=True)
        docker_output(
            [
                "run",
                "--rm",
                "--entrypoint",
                "filebrowser",
                *mount_args,
                image,
                "-d",
                database,
                "-c",
                config_file,
                "users",
                "update",
                username,
                "--password",
                password,
            ],
            combine_stderr=True,
        )
    finally:
        if was_running:
            docker_output(["start", container], combine_stderr=True)

    return FilebrowserLogin(username=username, password=password)


def format_login(
    login: FilebrowserLogin,
    *,
    container: str,
    ports: Sequence[str],
    output_format: str,
) -> str:
    if output_format == "json":
        return json.dumps(
            {
                "container": container,
                "username": login.username,
                "password": login.password,
                "published_ports": list(ports),
            },
            indent=2,
        )

    if output_format == "shell":
        lines = [
            f"FILEBROWSER_CONTAINER={shlex.quote(container)}",
            f"FILEBROWSER_USERNAME={shlex.quote(login.username)}",
            f"FILEBROWSER_PASSWORD={shlex.quote(login.password)}",
        ]
        if ports:
            lines.append(f"FILEBROWSER_PUBLISHED_PORTS={shlex.quote(' '.join(ports))}")
        return "\n".join(lines)

    lines = [
        "File Browser login",
        f"container: {container}",
        f"username: {login.username}",
        f"password: {login.password}",
    ]
    if ports:
        lines.append("published ports: " + ", ".join(ports))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        project = args.label_project or current_container_project()
        container = args.container or resolve_service_container(
            args.service,
            project=project,
        )
        display_container = container_display_name(container)
        if args.reset_password:
            password = args.password or generate_password()
            login = reset_filebrowser_password(
                container,
                username=args.username,
                password=password,
                database=args.database,
                config_file=args.config_file,
            )
            print(
                format_login(
                    login,
                    container=display_container,
                    ports=published_ports(container),
                    output_format=args.format,
                )
            )
            return 0

        if args.password:
            raise RuntimeError("--password requires --reset-password")

        logs = docker_output(["logs", container], combine_stderr=True)
        login = find_filebrowser_login(logs)
        if login is None:
            print(
                (
                    "No generated File Browser password was found in Docker "
                    f"logs for {display_container}.\n\n"
                    "File Browser prints the auto-generated admin password "
                    "only once, when it creates a new database. If Docker log "
                    "rotation removed that line, the clear-text password "
                    "cannot be recovered from the database.\n\n"
                    "Run this helper with --reset-password to set and print a "
                    "fresh password."
                ),
                file=sys.stderr,
            )
            return 2
        print(
            format_login(
                login,
                container=display_container,
                ports=published_ports(container),
                output_format=args.format,
            )
        )
        return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
