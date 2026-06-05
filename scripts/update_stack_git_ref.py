#!/usr/bin/env python3
"""Update the shared VBOGS checkout in the current stack repository volume."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


DEFAULT_REPO_DIR = "/workspace/VBOGS"


def git_update_script(git_ref: str, repo_dir: str) -> str:
    return f"""
set -eu
ref={shlex.quote(git_ref)}
repo_dir={shlex.quote(repo_dir)}
cd "$repo_dir"
if [ ! -d .git ]; then
  echo "No Git repository found in $repo_dir" >&2
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
            "Check out a Git branch, tag, or commit in the shared VBOGS "
            "repository volume mounted at /workspace/VBOGS."
        )
    )
    parser.add_argument("git_ref", help="Branch, tag, or commit to check out.")
    parser.add_argument(
        "--repo-dir",
        default=DEFAULT_REPO_DIR,
        help=f"Repository path in the shared volume. Defaults to {DEFAULT_REPO_DIR}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without running git.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = ["sh", "-lc", git_update_script(args.git_ref, args.repo_dir)]
    print(f"Git ref: {args.git_ref}")
    print(f"Repo dir: {args.repo_dir}")
    print("+ " + shlex.join(command))
    if args.dry_run:
        return 0
    return subprocess.run(command).returncode


if __name__ == "__main__":
    sys.exit(main())
