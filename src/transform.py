# transforms.py
#  by: mikasenghaas

import torch
from torchvision import transforms

from config import (
    MEAN,
    STD,
)


class ImageTransformer:
    """
    Class to transform images to the format required by FinetunedImageClassifier.

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
                transforms.Resize((224, 224)),
                transforms.Lambda(self.normalise),
                transforms.Normalize(MEAN, STD, inplace=True),
            ]
        )

    def normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float() / 255.0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transform(tensor)
