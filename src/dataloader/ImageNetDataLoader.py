from typing import Tuple, Union

import torch
import torchvision.transforms as transforms
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

from .. import utils
from ..registry import Registry


@Registry.register(prefix="data")
class ImageNetDataLoader(DataLoader):
    """PyTorch DataLoader for the ImageNet dataset.

    This DataLoader class loads the 2012 ImageNet classification dataset
    by acting as a wrapper class for the PyTorch ImageNet Dataset class.

    """

    def __init__(
        self,
        root: str,
        split: str,
        image_size: Union[int, Tuple[int, int]] = (256, 256),
        *args,
        **kwargs,
    ) -> None:
        """ImageNetDataLoader initialization.

        Args:
            root (str): The root directory of the dataset.
            split (str): The dataset split mode. Expected to be one of
              ["train", "val"].
            image_size (int | tuple): The image size (height, width) to be
              processed. All images will be resized to match the given
              dimensions. If the given image size is a single integer, it is
              assumed to have equal height and width values.

        Raises:
            AssertionError: When the split variable is unknown.

        """
        if split not in ["train", "val"]:
            raise AssertionError(f"Invalid dataset split mode {split}!")

        self.split = split
        self.image_size = (
            (image_size, image_size) if type(image_size) is int else image_size
        )

        # TODO: Add more augmentation methods
        self.augment = transforms.Compose(
            [
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
            ]
        )

        self.dataset = ImageNet(
            root=root,
            split=split,
            transform=self.transforms,
        )

        super(ImageNetDataLoader, self).__init__(
            dataset=self.dataset,
            collate_fn=self.collate_fn,
            *args,
            **kwargs,
        )

    def transforms(self, image: Image) -> torch.Tensor:
        image_tensor = utils.pil_to_tensor(image)
        if self.split == "train":
            image_tensor = self.augment(image_tensor)

        # TODO: Add pixel normalization
        image_tensor = utils.resize_tensor(image_tensor, size=self.image_size)
        return image_tensor

    def collate_fn(self, batch):
        images = torch.cat([image for (image, _) in batch])
        targets = torch.tensor([target for (_, target) in batch])
        return (images, targets)

    def __repr__(self) -> str:
        information = {
            "Name": f"{self.__class__.__name__}/{self.split}",
            "Module": f"{self.__class__.__module__}",
            "Location": hex(id(self)),
            "Number of Samples": len(self),
            "Number of Workers": self.num_workers,
        }

        return "\n\t".join(
            [f"{key}: {value}" for key, value in information.items()],
        )
