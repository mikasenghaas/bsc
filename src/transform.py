# transforms.py
#  by: mikasenghaas

from torchvision import transforms
from config import *

class ImageTransformer():
    def __init__(self) -> None:
        self.transform = transforms.Compose([ 
            transforms.Resize((224, 224)),
            transforms.Lambda(self.normalise),
            transforms.Normalize(MEAN, STD, inplace=True)
            ]) # pyright: ignore

    def normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float() / 255.0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transform(tensor)
