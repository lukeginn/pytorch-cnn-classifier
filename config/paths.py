from pathlib import Path
from enum import Enum


class Paths(Enum):
    BASE_PATH = Path().resolve()

    CONFIG_PATH = BASE_PATH / "config"
    CONFIG_FILE_PATH = CONFIG_PATH / "config.yaml"

    INPUTS_PATH = BASE_PATH / "inputs"
    OUTPUTS_PATH = BASE_PATH / "outputs"


def create_directories():
    for path in Paths:
        if path.value.suffix == "":  # Check if it's a directory (no suffix)
            path.value.mkdir(parents=True, exist_ok=True)
