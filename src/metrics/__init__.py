import torch

from ..registry import Registry


@Registry.register(prefix="metrics")
def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    batch_size = pred.shape[0]
    assert batch_size == target.shape[0], "Batch sizes do not match!"

    pred = pred.argmax(dim=1)
    return (torch.sum(pred == target) / batch_size).item()
