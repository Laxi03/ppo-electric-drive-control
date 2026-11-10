"""Resolve user artifacts separately from installed source code."""
import os
from pathlib import Path


def workspace_root() -> Path:
    """Use the working directory unless an explicit workspace is configured."""
    return Path(os.environ.get("RLLIB_CONTROL_WORKSPACE") or Path.cwd()).expanduser().resolve()
