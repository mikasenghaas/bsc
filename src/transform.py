# transforms.py
#  by: mikasenghaas

import torch
from torchvision import transforms

from config import (
    MEAN,
    STD,
)


class FrameTransformer:
    """
    Class to transform images to the format required by ImageClassifier and
    VideoClassifier.

    Does the following:
        - Resizes the image to 224x224
        - Normalises the image to the range [0, 1]
        - Normalises the image to the range [-1, 1] using the mean and standard
          deviation of the ImageNet dataset

    Attributes:
        transform (torchvision.transforms.Compose): Compose transforms

    Methods:
        normalise (torch.Tensor) -> torch.Tensor: Normalise a tensor to range [0, 1].
        __call__ (torch.Tensor) -> torch.Tensor: Apply the transform to a tensor.
    """

    def __init__(self) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=False),
                transforms.Lambda(self.scale),
                transforms.Normalize(MEAN, STD, inplace=True),
            ]
        )

    def scale(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float() / 255.0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies a series of transformations (resizing, normalising), to a
        single tensor (C, H, W), or a batch of tensors (B, C, H, W), or a batch
        of sequences of tensors (B, T, C, H, W).

        Args:
            tensor (torch.Tensor): Tensor to transform

        Returns:
            torch.Tensor: Transformed tensor
        """
        msg = "Tensor must be of shape (B, T, C, H, W), or (B, C, H, W), or (C, H, W)"
        assert tensor.ndim in [3, 4, 5], msg
        transformed = False
        if tensor.ndim == 5:
            B, T, C, H, W = tensor.shape
            tensor = tensor.view(B * T, C, H, W)
            transformed = True

        transformed_tensor = self.transform(tensor)

        if transformed:
            return transformed_tensor.view(B, T, C, H, W)

        return transformed_tensor
