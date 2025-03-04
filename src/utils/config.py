import yaml
import logging as logger


def read_config(config_file_path):
    # Load configuration from config.yaml
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
    config = ConfigDict(config)
    logger.info("Configuration loaded successfully")
    return config


class ConfigDict(dict):
    def __getattr__(self, item):
        value = self.get(item)
        if isinstance(value, dict):
            return ConfigDict(value)
        return value
