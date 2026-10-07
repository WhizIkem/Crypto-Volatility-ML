from pathlib import Path

import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """
    Load the project YAML configuration file.

    Parameters
    ----------
    config_path :
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration as a dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config
