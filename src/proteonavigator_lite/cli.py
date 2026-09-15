"""Command-line launcher for the local Chainlit application."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    app = Path(__file__).parents[2] / "app.py"
    environment = dict(os.environ)
    # Some clusters export DEBUG=release for unrelated software; Chainlit parses
    # DEBUG as a boolean before it imports the application.
    environment["DEBUG"] = "false"
    return subprocess.call(
        [sys.executable, "-m", "chainlit", "run", str(app)], env=environment
    )
