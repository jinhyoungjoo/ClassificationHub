import importlib
import os
from typing import Dict

import torch

from ..logger import Logger
from ..registry import Registry
from ..utils import import_modules

# Scan all files in directory to add modules to registry.
for module in import_modules(os.path.dirname(__file__)):
    importlib.import_module(f"{__name__}.{module}")


def build_model(config: Dict) -> torch.nn.Module:
    """Build model using configuration options.

    Args:
        config (Dict): The configuration options for the model. The dictionary
          must include a name field and an optional args field. The name field
          value is used to search the corresponding object from the registry.

    Returns:
        nn.Module: A model object retrieved from the registry.

    """
    try:
        name = config["name"]
    except KeyError:
        Logger.error("Missing name field in dataloader configuration!")
        exit(-1)

    name = config.get("name", "")
    parameters = config.get("args", {})

    try:
        model = Registry.get(
            prefix="model",
            name=name,
        )(**parameters)
        Logger.info("Model build success." f"\n{model}")

    except KeyError as e:
        Logger.error(f"Error importing {name}\n{e}")
        exit(-1)

    return model


def build_optimizer(params, config: Dict) -> torch.optim.Optimizer:
    """Build optimizer using configuration options.

    Args:
        params: PyTorch optimizers require the model parameters to optimize.
          Use model.parameters() to pass in as an argument to the optimizer.
        config (Dict): The configuration options for the optimizer. The
          dictionary must include a name field and an optional args field.
          The name field value is used to search the corresponding object
          from the registry.

    Returns:
        optim.Optimizer: An optimizer object retrieved from the registry.

    """
    try:
        name = config["name"]
    except KeyError:
        Logger.error("Missing name field in dataloader configuration!")
        exit(-1)

    name = config.get("name", "")
    parameters = config.get("args", {})

    try:
        optimizer = Registry.get_from_module(
            name,
            torch.optim,
            params=params,
            **parameters,
        )
        Logger.info("Optimizer build success." f"\n{optimizer}")

    except AttributeError as e:
        Logger.error(f"Error importing {name}\n{e}")
        exit(-1)

    return optimizer
