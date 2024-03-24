import os
from abc import abstractmethod

import torch
from torch.backends import mps

from ..logger import Logger

DEFAULT_OPTIONS = {
    "num_epochs": 1,
    "log_step": 5,
    "save_period": 1,
}


class BaseTrainer:
    """Base Trainer Class"""

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_dataloader,
        validation_dataloader,
        **kwargs,
    ) -> None:
        """Initialization of BaseTrainer

        Args:
            model: The model that will be trained.
            optimizer: The optimizer that will be used.
            criterion: The loss function the model will be trained on.
            train_dataloader: The dataloader for the train set.
            validation_dataloader: The dataloader for the validation set.

        Kwargs:
            device: Specify the device to use for training.

        """
        self.device = kwargs.get(
            "device",
            torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if mps.is_available() else "cpu"
            ),
        )

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.base_dir = Logger.base_dir()
        self.checkpoint_dir = Logger.checkpoint_dir()

        self.train_config = kwargs.get("train_config", {})
        self.validation_config = kwargs.get("validation_config", {})

        for option, value in DEFAULT_OPTIONS.items():
            if option in self.train_config:
                continue

            self.train_config[option] = value
            Logger.warn(
                f"Argument {option} is not specified in configuration. "
                f"Using default value {value}."
            )

        self.num_epochs = self.train_config["num_epochs"]
        self.start_epoch, self.total_iters = 1, 0
        self.load()

        self.log_step = self.train_config["log_step"]
        self.save_period = self.train_config["save_period"]

        self.validation_metrics = self.validation_config.get("metrics", [])

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.model.train()
            self.train_one_epoch(epoch)

            with torch.no_grad():
                self.model.eval()

                if self.validation_config is not None:
                    self.validate(epoch)

                if (epoch + 1) % self.save_period == 0:
                    self.save(epoch)

    @abstractmethod
    def train_one_epoch(self, epoch: int):
        raise NotImplementedError(
            "Function 'train_one_epoch' not implemented!",
        )

    @abstractmethod
    def validate(self, epoch: int):
        raise NotImplementedError("Function 'validate' not implemented!")

    def save(self, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(self.checkpoint_dir, f"epoch={epoch}.ckpt"),
        )

    def load(self) -> None:
        try:
            checkpoint_path = max(
                [
                    os.path.join(self.checkpoint_dir, checkpoint)
                    for checkpoint in os.listdir(self.checkpoint_dir)
                ],
                key=os.path.getctime,
            )
        except FileNotFoundError as e:
            Logger.error(f"Checkpoint directory not found! {e}")
            exit()
        except ValueError:
            Logger.info(
                "No checkpoint file found to load. "
                "Starting training procedure from the beginning."
            )
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint["epoch"] + 1