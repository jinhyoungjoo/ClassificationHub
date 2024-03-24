from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchmetrics import Accuracy


class ClassificationLoss(nn.Module):
    def __init__(self, module, *args, **kwargs) -> None:
        super(ClassificationLoss, self).__init__(*args, **kwargs)
        self.module = module
        self.name = self.abbreviate_name(module.__class__.__name__)

        self.accuracy = Accuracy(
            task="multiclass",
            num_classes=kwargs.get("num_classes", 1000),
        )

    def forward(
        self,
        pred,
        target,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        loss = self.module.forward(pred, target, *args, **kwargs)
        accuracy = self.accuracy(
            pred.softmax(dim=-1),
            target,
        )

        return loss, {
            self.name: loss,
            "acc": accuracy,
        }

    def __repr__(self):
        return self.module.__repr__()

    @staticmethod
    def abbreviate_name(name: str) -> str:
        name = name.replace("Loss", "")
        name = "".join([c for c in name if c.isupper()]).lower()
        return name + "_loss"
