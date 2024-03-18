import argparse
import pathlib
from typing import Any, Dict

import yaml

__all__ = ["parse_args"]


def parse_args() -> Dict[str, Any]:
    """Parse command-line arguments.

    Parse command-line arguments using ArgumentParser. A configuration file
    is required to execute.

    Returns:
        Dict: A dictionary of configuration values.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="Configuration file (.yaml file).",
    )

    args = vars(parser.parse_args())
    config_file = args.pop("config")
    args = {**parse_yaml(config_file), **args, "config_file": config_file}
    return args


def parse_yaml(file: pathlib.Path) -> Dict[str, Any]:
    """Parse given yaml file.

    Args:
        file (pathlib.Path): Filepath to the target yaml file to parse.

    Returns:
        Dict: A dictionary of key-value pairs extracted from the yaml file.

    Raises:
        SystemExit: When the yaml filepath is invalid or cannot be read.

    """
    try:
        with open(file, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise SystemExit(f"Error reading configuration file {file}!\n{e}")
    except Exception as e:
        raise SystemExit(f"Invalid configuration file {file}!\n{e}")

    return data
