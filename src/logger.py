import logging
import os
import shutil
from time import localtime, strftime
from typing import Dict

from torch.utils.tensorboard.writer import SummaryWriter


class Logger(object):
    """Singleton pattern logger class."""

    instance_ = None

    project_name_ = ""
    run_id_ = ""
    base_dir_ = ""
    verbosity_ = 0

    file_logger_ = None
    tb_logger_ = None

    def __new__(
        cls,
        project_name: str,
        config: Dict,
    ):
        """Logger instantiation.

        Args:
            project_name (str): The name of the project.
            config (Dict): Logger configuration. The following keys are used
              for configuration: ["verbosity", "use_tensorboard"].

        """
        if cls.instance_ is not None:
            return cls.instance_

        cls.instance_ = super(Logger, cls).__new__(cls)

        # Instance information
        cls.project_name_ = project_name
        cls.base_dir_ = os.path.join("experiments", project_name)
        cls.verbosity_ = config.get("verbosity", 0)
        cls.run_id_ = strftime("%Y%m%d%H%M%S", localtime())

        # Create directories
        os.makedirs(cls.log_dir(), exist_ok=True)
        os.makedirs(cls.checkpoint_dir(), exist_ok=True)

        cls.file_logger_ = Logger.build_filelogger()

        if config.get("use_tensorboard", False):
            tb_logs_path = os.path.join(cls.base_dir_, "tb_logs")
            os.makedirs(tb_logs_path, exist_ok=True)
            cls.tb_logger_ = SummaryWriter(log_dir=tb_logs_path)

        return cls.instance_

    @classmethod
    def clean(cls, remove_logs: bool = True):
        """Clean logger instance.

        Args:
            remove_logs (bool): Delete all log files associated with the
              cuurent logger.
        """
        cls.instance_ = None

        if remove_logs:
            shutil.rmtree(cls.base_dir_)

    @classmethod
    def build_filelogger(cls):
        """Build file logger."""
        logger = logging.getLogger()
        logger.setLevel(cls.verbosity_)

        logger_path = os.path.join(cls.base_dir_, "logs", f"{cls.run_id_}.log")
        logger_format = (
            f"%(asctime)s[{cls.project_name_}]" "[%(levelname)s] %(message)s"
        )

        logging.basicConfig(
            format=logger_format,
            datefmt="[%Y/%m/%d %H:%M:%S]",
            handlers=[
                logging.FileHandler(str(logger_path), mode="a"),
                logging.StreamHandler(),
            ],
        )
        return logger

    @classmethod
    def info(cls, message):
        if cls.file_logger_ is None:
            return

        cls.file_logger_.info(message)

    @classmethod
    def warn(cls, message):
        if cls.file_logger_ is None:
            return

        cls.file_logger_.warn(message)

    @classmethod
    def error(cls, message):
        if cls.file_logger_ is None:
            return

        cls.file_logger_.error(message)

    @classmethod
    def scalar(cls, tag, value, *args, **kwargs):
        if cls.tb_logger_ is None:
            return

        cls.tb_logger_.add_scalar(tag, value, *args, **kwargs)

    @classmethod
    def base_dir(cls) -> str:
        return cls.base_dir_

    @classmethod
    def log_dir(cls) -> str:
        return os.path.join(cls.base_dir(), "logs")

    @classmethod
    def checkpoint_dir(cls) -> str:
        return os.path.join(cls.base_dir(), "checkpoints")
