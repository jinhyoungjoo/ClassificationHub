import os
import shutil
from glob import glob
from typing import List, Union

import numpy as np
import pyfiglet
import torch
import torch.backends.cudnn as cudnn
import torchvision

from ..logger import Logger

__all__ = ["setup_project", "import_modules"]


def setup_project(args):
    project_name = args.get("name", None)
    if project_name is None:
        raise KeyError("Project name required in configuration file!")

    logger_config = args.get("logger", {})
    logger = Logger(project_name, logger_config)
    logger.info(welcome_message(project_name))

    shutil.copy(
        args["config_file"],
        f"experiments/{project_name}/{project_name}.yaml",
    )

    seed = args.get("seed", None)
    if seed is not None:
        set_seed(seed)
        logger.info(f"Seed set to {seed}.")


def import_modules(directory: str) -> List[str]:
    """Get list of submodules inside directory.

    Get all files within a submodule excluding the __init__.py file. This list
    can be used to import all the files neccesary to build a registry of the
    module by dynamically importing files without manual specification.

    Args:
        directory (str): The directory the function scans to find files to
          import.

    Returns:
        List[str]: A list of modules

    Examples:
        In __init__.py:
        >>> for module in import_modules(os.path.dirname(__file__)):
        ...     importlib.import_module(f"{__name__}.{module}")

    """
    modules = []
    for x in glob(os.path.join(directory, "*.py")):
        basename = os.path.basename(x)[:-3]
        if basename == "__init__":
            continue

        modules.append(basename)

    return modules


def welcome_message(project_name: str) -> str:
    """Print a welcome message for the project.

    Args:
        project_name (str): The name of the project.

    """
    os.system("clear")

    message = "\n"
    message += pyfiglet.figlet_format(
        f"Project\nClassificationHub\n- {project_name.upper()} -",
        font="slant",
        justify="center",
        width=100,
    )

    message += (
        "\nVersion Information: "
        f"\n\tPyTorch: {torch.__version__}"
        f"\n\tTorchVision: {torchvision.__version__}"
    )

    return message


def set_seed(seed: Union[int, None]) -> None:
    if seed is None:
        return

    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(seed)
