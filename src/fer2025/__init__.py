"""FER2025 real-time facial emotion recognition package."""

from importlib import resources
from pathlib import Path
from typing import Any, Dict

import yaml


def load_default_config() -> Dict[str, Any]:
    """Load the default configuration bundled with the package."""
    with resources.files(__package__).joinpath("config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_path() -> Path:
    """Return the path to the packaged default configuration file."""
    return Path(resources.files(__package__).joinpath("config.yaml"))


__all__ = ["load_default_config", "get_config_path"]
