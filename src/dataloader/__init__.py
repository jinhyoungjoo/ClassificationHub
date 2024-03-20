import importlib
import os
from typing import Dict

from torch.utils.data import DataLoader

from ..logger import Logger
from ..registry import Registry
from ..utils import import_modules

# Scan all files in directory to add modules to registry.
for module in import_modules(os.path.dirname(__file__)):
    importlib.import_module(f"{__name__}.{module}")


def build_dataloader(config: Dict) -> DataLoader:
    """Build dataloader using configuration options.

    Args:
        config (Dict): The configuration options for the dataloader. The
          dictionary must include a name field and an optional args field.
          The name field value is used to search the corresponding object
          from the registry.

    Returns:
        DataLoader: A dataloader object retrieved from the registry.

    """
    try:
        name = config["name"]
    except KeyError:
        Logger.error("Missing name field in dataloader configuration!")
        exit(-1)

    parameters = config.get("args", {})

    try:
        dataloader = Registry.get(
            prefix="data",
            name=name,
        )(**parameters)

    except KeyError as e:
        Logger.error(f"Error generating {name}\n{e}")
        exit(-1)

    Logger.info(f"Dataloader build success.\n{dataloader}")

    return dataloader
