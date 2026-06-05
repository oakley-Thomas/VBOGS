#!/usr/bin/env python3
"""Bootstrap VBOGS into the shared stack repository volume."""

from __future__ import annotations

import argparse
import getpass
import os
import shlex
import subprocess
import sys
import urllib.parse


DEFAULT_REPO_DIR = "/workspace/VBOGS"
DEFAULT_REPOSITORY = "oakley-Thomas/VBOGS"
DEFAULT_REF = "main"


def normalize_repository_url(repository: str) -> str:
    value = repository.strip()
    if not value:
        raise ValueError("repository cannot be empty")

    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    elif "://" not in value:
        value = "https://github.com/" + value.strip("/")

    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "https" or parsed.netloc != "github.com":
        raise ValueError("repository must be a GitHub https URL or owner/name")

    if not value.endswith(".git"):
        value = value.rstrip("/") + ".git"
    return value


def git_bootstrap_script(
    *,
    repo_url: str,
    git_ref: str,
    repo_dir: str,
) -> str:
    return f"""
set -eu
repo_url={shlex.quote(repo_url)}
ref={shlex.quote(git_ref)}
repo_dir={shlex.quote(repo_dir)}

askpass="$(mktemp /tmp/vbogs-git-askpass.XXXXXX)"
cleanup() {{
  rm -f "$askpass"
}}
trap cleanup EXIT HUP INT TERM
cat > "$askpass" <<'VBOGS_ASKPASS'
#!/bin/sh
case "$1" in
  *Username*) printf '%s\\n' "${{VBOGS_GITHUB_USER:?}}" ;;
  *Password*|*password*) printf '%s\\n' "${{VBOGS_GITHUB_TOKEN:?}}" ;;
  *) printf '\\n' ;;
esac
VBOGS_ASKPASS
chmod 700 "$askpass"

export GIT_ASKPASS="$askpass"
export GIT_TERMINAL_PROMPT=0

mkdir -p "$repo_dir"
cd "$repo_dir"
if [ ! -d .git ]; then
  git init
  git remote add origin "$repo_url"
elif git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$repo_url"
else
  git remote add origin "$repo_url"
fi

git fetch --tags --prune origin
if git show-ref --verify --quiet "refs/remotes/origin/${{ref}}"; then
  git checkout -B "$ref" "origin/$ref"
elif git show-ref --verify --quiet "refs/tags/${{ref}}"; then
  git checkout "$ref"
else
  git checkout "$ref"
fi

git submodule sync --recursive
git submodule update --init --recursive
git remote set-url origin "$repo_url"
git status --short --branch
""".strip()


def prompt_secret(args: argparse.Namespace) -> tuple[str, str]:
    github_user = args.github_user or os.environ.get("VBOGS_GITHUB_USER", "")
    if not github_user:
        github_user = input("GitHub username: ").strip()
    github_token = os.environ.get(args.github_token_env, "")
    if not github_token:
        github_token = getpass.getpass("GitHub token: ")

    if not github_user:
        raise ValueError("GitHub username is required")
    if not github_token:
        raise ValueError("GitHub token is required")
    return github_user, github_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch VBOGS from GitHub into the current container's shared "
            "repository volume. Server stacks mount this volume into every "
            "VBOGS runtime service at /workspace/VBOGS."
        )
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("VBOGS_GIT_REPOSITORY", DEFAULT_REPOSITORY),
        help=(
            "GitHub repository as owner/name or https URL. Defaults to "
            f"{DEFAULT_REPOSITORY}."
        ),
    )
    parser.add_argument(
        "--ref",
        default=os.environ.get("VBOGS_GIT_REF", DEFAULT_REF),
        help=f"Branch, tag, or commit to check out. Defaults to {DEFAULT_REF}.",
    )
    parser.add_argument(
        "--repo-dir",
        default=DEFAULT_REPO_DIR,
        help=f"Repository path in the shared volume. Defaults to {DEFAULT_REPO_DIR}.",
    )
    parser.add_argument(
        "--github-user",
        default="",
        help="GitHub username. Defaults to prompting interactively.",
    )
    parser.add_argument(
        "--github-token-env",
        default="VBOGS_GITHUB_TOKEN",
        help=(
            "Environment variable to read the GitHub token from before prompting. "
            "Defaults to VBOGS_GITHUB_TOKEN."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the target repo path without running git.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        repo_url = normalize_repository_url(args.repository)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.dry_run:
            github_user = args.github_user or os.environ.get("VBOGS_GITHUB_USER", "dry-run")
            github_token = os.environ.get(args.github_token_env, "dry-run")
        else:
            github_user, github_token = prompt_secret(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Repository: {repo_url}")
    print(f"Git ref: {args.ref}")
    print(f"Repo dir: {args.repo_dir}")
    print("GitHub token: <hidden>")

    command = [
        "sh",
        "-lc",
        git_bootstrap_script(
            repo_url=repo_url,
            git_ref=args.ref,
            repo_dir=args.repo_dir,
        ),
    ]
    print("+ " + shlex.join(command))
    if args.dry_run:
        return 0

    env = {
        **os.environ,
        "VBOGS_GITHUB_USER": github_user,
        "VBOGS_GITHUB_TOKEN": github_token,
    }
    return subprocess.run(command, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
