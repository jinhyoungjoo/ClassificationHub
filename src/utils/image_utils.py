import PIL.Image as Image
import torch
import torchvision.transforms.functional as F


def pil_to_tensor(
    pil_image: Image.Image,
    expand_dims: bool = True,
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """Convert a PIL image to a PyTorch tensor.

    Args:
        pil_image (PIL.Image): The PIL image to transform.
        expand_dims (bool): If true, an additional dimension is added to the
          tensor.
        dtype (torch.dtype): The target tensor's data type.

    Returns:
        torch.Tensor: The converted PIL image in PyTorch tensor format.

    """
    tensor_image = F.pil_to_tensor(pil_image).to(dtype)

    if expand_dims:
        tensor_image = tensor_image.unsqueeze(0)

    return tensor_image
