import collections
from typing import Any, List

import torch
import torch.nn as nn
from torchinfo import summary

from ..registry import Registry


@Registry.register(prefix="model")
class VGGNet(nn.Module):
    """PyTorch Implmentation of VGG-Net

    A PyTorch implementation of the VGG-Net model from the paper `Very Deep
    Convolutional Networks for Large-Scale Image Recognition`.

    """

    def __init__(
        self,
        in_channels: int,
        model_configuration: str,
        *args,
        **kwargs,
    ) -> None:
        """Initialization of VGGNet.

        Args:
            in_channels (int): Number of channels of the input image.
            model_configuration (str): The model configuration key.

        Raises:
            AssertionError: When the model configuration is unknown.

        """
        super(VGGNet, self).__init__(*args, **kwargs)

        assert model_configuration in VGGNET_MODEL_CONFIGURATIONS.keys(), (
            f"Invalid model configuration {model_configuration}!"
            f"Available configurations are \
                    {VGGNET_MODEL_CONFIGURATIONS.keys()}"
        )

        configuration = VGGNET_MODEL_CONFIGURATIONS[model_configuration]

        self.convolutional_layers = self.build_convolutional_layers(
            in_channels,
            configuration,
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.Softmax(dim=1),
        )

    @property
    def input_details(self) -> Any:
        return (3, 3, 224, 224)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of VGG-Net

        Args:
            x (torch.Tensor): The input image. Shape expected to be as defined
              in `self.input_details`.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = self.convolutional_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x

    def __repr__(self) -> str:
        return str(summary(self, self.input_details, verbose=0))

    def build_convolutional_layers(
        self,
        in_channels: int,
        configuration: List,
    ) -> nn.Module:
        layers, out_channels = [], 0

        for idx, layer_configuration in enumerate(configuration):
            layer = []
            for module in layer_configuration:
                name = module[0]
                if name == "conv":
                    kernel_size, out_channels, repeat_count = module[1:]
                    layer.append(
                        ConvBlock(
                            in_channels,
                            out_channels,
                            kernel_size,
                            repeat_count=repeat_count,
                        )
                    )
                    in_channels = out_channels

                elif name == "lrn":
                    # Parameters from AlexNet
                    layer.append(nn.LocalResponseNorm(size=5, k=2))

            layer.append(nn.MaxPool2d(kernel_size=2))

            layers.append(
                (f"conv_{idx}", nn.Sequential(*layer)),
            )

        return nn.Sequential(collections.OrderedDict(layers))


class ConvBlock(nn.Module):
    """Basic Convolution Block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        same_size: bool = True,
        repeat_count: int = 1,
        *args,
        **kwargs,
    ) -> None:
        """Initialization of ConvBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): The size of the convolution kernel.
            same_size (bool): If true, the padding value will be set based on
              the kernel_size to remain the image dimensions.
            repeat_count (int): Number of ConvBlocks to chain together. Every
              ConvBlock except the first block will have both in_channels and
              out_channels as out_channels.

        """
        super(ConvBlock, self).__init__()

        if same_size:
            kwargs["padding"] = (kernel_size - 1) // 2

        layers = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    *args,
                    **kwargs,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        ]

        for _ in range(repeat_count - 1):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        *args,
                        **kwargs,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ConvBlock.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The output tensor.

        """
        return self.layers(x)


VGGNET_MODEL_CONFIGURATIONS = {
    "A": [
        [("conv", 3, 64, 1)],
        [("conv", 3, 128, 1)],
        [("conv", 3, 256, 2)],
        [("conv", 3, 512, 2)],
        [("conv", 3, 512, 2)],
    ],
    "A-LRN": [
        [("conv", 3, 64, 1), ("lrn")],
        [("conv", 3, 128, 1)],
        [("conv", 3, 256, 2)],
        [("conv", 3, 512, 2)],
        [("conv", 3, 512, 2)],
    ],
    "B": [
        [("conv", 3, 64, 2)],
        [("conv", 3, 128, 2)],
        [("conv", 3, 256, 2)],
        [("conv", 3, 512, 2)],
        [("conv", 3, 512, 2)],
    ],
    "C": [
        [("conv", 3, 64, 2)],
        [("conv", 3, 128, 2)],
        [("conv", 3, 256, 2), ("conv", 1, 256, 1)],
        [("conv", 3, 512, 2), ("conv", 1, 512, 1)],
        [("conv", 3, 512, 2), ("conv", 1, 512, 1)],
    ],
    "D": [
        [("conv", 3, 64, 2)],
        [("conv", 3, 128, 2)],
        [("conv", 3, 256, 3)],
        [("conv", 3, 512, 3)],
        [("conv", 3, 512, 3)],
    ],
    "E": [
        [("conv", 3, 64, 2)],
        [("conv", 3, 128, 2)],
        [("conv", 3, 256, 4)],
        [("conv", 3, 512, 4)],
        [("conv", 3, 512, 4)],
    ],
}
