import importlib
import os
from typing import Dict

import torch.nn as nn

from ..logger import Logger
from ..registry import Registry
from ..utils import import_modules

# Scan all files in directory to add modules to registry.
for module in import_modules(os.path.dirname(__file__)):
    importlib.import_module(f"{__name__}.{module}")


def build_criterion(config: Dict) -> nn.Module:
    """Build criterion using configuration options.

    Args:
        config (Dict): The configuration options for the criterion. The
          dictionary must include a name field and an optional args field.
          The name field value is used to search the corresponding object
          from the registry.

    Returns:
        nn.Module: A criterion object retrieved from the registry.

    """
    try:
        name = config["name"]
    except KeyError:
        Logger.error("Missing name field in criterion configuration!")
        exit(-1)

    parameters = config.get("args", {})

    try:
        criterion = Registry.get_from_module(
            name,
            nn,
            **parameters,
        )
        Logger.info("Criterion build success." f"\n{criterion}")

    except AttributeError as e:
        Logger.error(f"Error importing {name}\n{e}")
        exit(-1)

    return criterion
