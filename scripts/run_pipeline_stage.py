#!/usr/bin/env python3
"""Run one VBOGS stage in its own process group inside a compute container.

This wrapper is used by the web scheduler so cancellation targets only the
current job, never the shared Torch/JAX service container.
"""

from __future__ import annotations

import argparse
import errno
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("A stage command is required")
    # ``docker exec`` commonly starts the command as a process-group leader.
    # POSIX rejects ``setsid`` for a group leader with EPERM, but that is
    # already a safe, isolated group for the scheduler to signal.  Preserve
    # all other errors and record the actual group ID below.
    try:
        os.setsid()
    except PermissionError as exc:
        if exc.errno != errno.EPERM or os.getpgrp() != os.getpid():
            raise
    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.pid_file.write_text(f"{os.getpgrp()}\n", encoding="utf-8")
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
