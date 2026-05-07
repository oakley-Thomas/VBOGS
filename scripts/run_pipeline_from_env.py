#!/usr/bin/env python3

"""Launch the VBOGS pipeline runner from compose environment variables."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys


def main() -> None:
    cmd = [
        sys.executable,
        "scripts/run_drive_pipeline.py",
        "--config",
        os.environ.get("VBOGS_PIPELINE_CONFIG") or "pipeline_config.yaml",
        "--use-service-labels",
    ]

    drive = os.environ.get("VBOGS_DRIVE")
    if drive:
        cmd.extend(["--drive", drive])

    extra_args = os.environ.get("VBOGS_PIPELINE_ARGS", "")
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    print("+ " + shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
