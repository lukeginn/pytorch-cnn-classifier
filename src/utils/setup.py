import numpy as np
import logging as logger
import config.paths as paths
from src.utils.config import read_config
from src.utils.check_gpu import check_gpu
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Setup:
    config: Dict[str, any] = field(init=False)

    def __post_init__(self) -> None:
        logger.info("Setting up paths and configurations")
        paths.create_directories()
        self.config = read_config(config_file_path=paths.Paths.CONFIG_FILE_PATH.value)
        check_gpu()
        np.random.seed(self.config.random_state)
