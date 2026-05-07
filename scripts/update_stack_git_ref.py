#!/usr/bin/env python3
"""Update the VBOGS checkout in every running container in this compose stack."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shlex
import socket
import sys
import urllib.parse
from dataclasses import dataclass
from typing import Any


DEFAULT_SERVICES = ("vbogs-torch", "vbogs-jax", "vbogs-pipeline")
DEFAULT_DOCKER_SOCKET = "/var/run/docker.sock"
DEFAULT_REPO_DIR = "/workspace/VBOGS"


class UnixHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection variant that talks to Docker over a Unix socket."""

    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


class DockerAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class Container:
    id: str
    name: str
    service: str


class DockerClient:
    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        expect: tuple[int, ...] = (200,),
    ) -> Any:
        payload = None
        headers = {}
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        conn = UnixHTTPConnection(self.socket_path)
        try:
            conn.request(method, path, body=payload, headers=headers)
            response = conn.getresponse()
            raw = response.read()
        finally:
            conn.close()

        if response.status not in expect:
            detail = raw.decode("utf-8", errors="replace").strip()
            raise DockerAPIError(f"{method} {path} failed with HTTP {response.status}: {detail}")

        if not raw:
            return None
        content_type = response.getheader("Content-Type", "")
        if "application/json" in content_type:
            return json.loads(raw.decode("utf-8"))
        return raw

    def inspect_container(self, container_id: str) -> dict[str, Any]:
        return self.request("GET", f"/containers/{urllib.parse.quote(container_id)}/json")

    def list_containers(self, filters: dict[str, list[str]]) -> list[dict[str, Any]]:
        query = urllib.parse.urlencode({"filters": json.dumps(filters)})
        return self.request("GET", f"/containers/json?{query}")

    def create_exec(self, container_id: str, cmd: list[str], workdir: str) -> str:
        response = self.request(
            "POST",
            f"/containers/{urllib.parse.quote(container_id)}/exec",
            body={
                "AttachStdout": True,
                "AttachStderr": True,
                "Tty": False,
                "WorkingDir": workdir,
                "Cmd": cmd,
            },
            expect=(201,),
        )
        return response["Id"]

    def start_exec(self, exec_id: str) -> tuple[str, str]:
        raw = self.request(
            "POST",
            f"/exec/{urllib.parse.quote(exec_id)}/start",
            body={"Detach": False, "Tty": False},
        )
        if not isinstance(raw, bytes):
            return "", ""
        return demux_docker_stream(raw)

    def inspect_exec(self, exec_id: str) -> dict[str, Any]:
        return self.request("GET", f"/exec/{urllib.parse.quote(exec_id)}/json")


def demux_docker_stream(raw: bytes) -> tuple[str, str]:
    stdout = bytearray()
    stderr = bytearray()
    index = 0

    while index + 8 <= len(raw):
        stream_type = raw[index]
        size = int.from_bytes(raw[index + 4 : index + 8], "big")
        next_index = index + 8 + size
        if stream_type not in (1, 2) or next_index > len(raw):
            break
        chunk = raw[index + 8 : next_index]
        if stream_type == 1:
            stdout.extend(chunk)
        else:
            stderr.extend(chunk)
        index = next_index

    if index != len(raw):
        stdout = bytearray(raw)
        stderr = bytearray()

    return (
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def current_compose_project(client: DockerClient) -> str:
    container_id = os.environ.get("HOSTNAME", "")
    if not container_id:
        return ""
    try:
        details = client.inspect_container(container_id)
    except DockerAPIError:
        return ""
    labels = details.get("Config", {}).get("Labels") or {}
    return labels.get("com.docker.compose.project", "")


def resolve_container(
    client: DockerClient,
    *,
    service: str,
    project: str,
) -> Container:
    label_filters = [f"com.docker.compose.service={service}"]
    if project:
        label_filters.append(f"com.docker.compose.project={project}")

    matches = client.list_containers({"label": label_filters, "status": ["running"]})
    if len(matches) != 1:
        project_hint = f" in compose project {project!r}" if project else ""
        raise DockerAPIError(
            f"expected exactly one running container for service {service!r}{project_hint}; "
            f"found {len(matches)}"
        )

    match = matches[0]
    names = match.get("Names") or [match["Id"][:12]]
    return Container(
        id=match["Id"],
        name=names[0].lstrip("/"),
        service=service,
    )


def git_update_script(git_ref: str) -> str:
    quoted_ref = shlex.quote(git_ref)
    return f"""
set -eu
ref={quoted_ref}
if [ ! -d .git ]; then
  echo "No Git repository found in $(pwd)" >&2
  exit 2
fi
git fetch --tags origin
if git show-ref --verify --quiet "refs/heads/${{ref}}"; then
  git checkout "${{ref}}"
  if git show-ref --verify --quiet "refs/remotes/origin/${{ref}}"; then
    git pull --ff-only origin "${{ref}}"
  fi
elif git show-ref --verify --quiet "refs/remotes/origin/${{ref}}"; then
  git checkout -B "${{ref}}" "origin/${{ref}}"
else
  git checkout "${{ref}}"
fi
git submodule update --init --recursive
git status --short --branch
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check out a Git branch, tag, or commit in every running VBOGS service "
            "container in the current Docker Compose stack."
        )
    )
    parser.add_argument("git_ref", help="Branch, tag, or commit to check out.")
    parser.add_argument(
        "--services",
        nargs="+",
        default=list(DEFAULT_SERVICES),
        help=f"Compose services to update. Defaults to: {', '.join(DEFAULT_SERVICES)}.",
    )
    parser.add_argument(
        "--project",
        default="",
        help="Compose project name. Defaults to the project label on the current container.",
    )
    parser.add_argument(
        "--repo-dir",
        default=DEFAULT_REPO_DIR,
        help=f"Repository path inside each container. Defaults to {DEFAULT_REPO_DIR}.",
    )
    parser.add_argument(
        "--docker-socket",
        default=DEFAULT_DOCKER_SOCKET,
        help=f"Docker socket path. Defaults to {DEFAULT_DOCKER_SOCKET}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target containers and commands without running git.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.docker_socket):
        print(
            f"Docker socket not found at {args.docker_socket}. "
            "Mount /var/run/docker.sock into this container to update sibling containers.",
            file=sys.stderr,
        )
        return 2

    client = DockerClient(args.docker_socket)
    project = args.project or current_compose_project(client)
    containers = [
        resolve_container(client, service=service, project=project)
        for service in args.services
    ]

    command = ["sh", "-lc", git_update_script(args.git_ref)]
    project_label = project or "<any project>"
    print(f"Git ref: {args.git_ref}")
    print(f"Compose project: {project_label}")
    print(f"Repo dir: {args.repo_dir}")

    failed = False
    for container in containers:
        print(f"\n=== {container.service} ({container.name}) ===", flush=True)
        print("+ docker exec -w " + shlex.quote(args.repo_dir) + " " + container.name + " " + shlex.join(command))
        if args.dry_run:
            continue

        exec_id = client.create_exec(container.id, command, args.repo_dir)
        stdout, stderr = client.start_exec(exec_id)
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)

        details = client.inspect_exec(exec_id)
        exit_code = details.get("ExitCode")
        if exit_code != 0:
            failed = True
            print(
                f"{container.service} failed with exit code {exit_code}",
                file=sys.stderr,
            )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
