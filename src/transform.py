# transforms.py
#  by: mikasenghaas

from torchvision import transforms
from config import *


class ImageTransformer:
    """
    Class to transform images to the format required by FinetunedImageClassifier.

    Does the following:
        - Resizes the image to 224x224
        - Normalises the image to the range [0, 1]
        - Normalises the image to the range [-1, 1] using the mean and standard deviation of the ImageNet dataset

    Attributes:
        transform (torchvision.transforms.Compose): Compose of transforms to be applied to images.

    Methods:
        normalise (torch.Tensor) -> torch.Tensor: Normalise a tensor to the range [0, 1].
        __call__ (torch.Tensor) -> torch.Tensor: Apply the transform to a tensor.
    """

    def __init__(self) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Lambda(self.normalise),
                transforms.Normalize(MEAN, STD, inplace=True),
            ]
        )  # pyright: ignore

    def normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float() / 255.0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transform(tensor)
