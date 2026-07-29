#!/usr/bin/env python3
"""Start the VBOGS Web Experiment Console."""

from __future__ import annotations

import os

import uvicorn

from vbogs.web.app import create_app


if __name__ == "__main__":
    uvicorn.run(create_app(), host=os.environ.get("VBOGS_GUI_HOST", "0.0.0.0"), port=int(os.environ.get("VBOGS_GUI_PORT", "8090")))
