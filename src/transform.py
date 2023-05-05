# transforms.py
#  by: mikasenghaas
import warnings
import torch

warnings.filterwarnings("ignore", category=UserWarning)

from torchvision.transforms import (  # noqa
    ToTensor,
    Normalize,
    Resize,
    Lambda,
    Compose,
    CenterCrop,
)
from torchvision.transforms._transforms_video import (  # noqa
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (  # noqa
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)


class ImageTransform:
    def __init__(self, mean, std, crop_size) -> None:
        self.transform = Compose(
            [
                ToTensor(),
                Normalize(mean, std, inplace=True),
                Resize((crop_size, crop_size), antialias=False),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, do_transform, alpha=4):
        super().__init__()
        self.do_transform = do_transform
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        if not self.do_transform:
            return frames
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // 4
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


class VideoTransform:
    """
    VideoTransform is a wrapper for `torchvision.transforms.Compose` that applies
    a series of transformations to a video tensor.
    """

    def __init__(
        self, mean, std, side_size, crop_size, num_frames, sampling_rate, packway
    ):
        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(self.normalise),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size),
                    PackPathway(do_transform=packway, alpha=sampling_rate),
                ]
            ),
        )

    def normalise(self, video):
        return video / 255.0

    def __call__(self, video):
        return self.transform(video)


if __name__ == "__main__":
    transform = ImageTransform(0.480, 0.225, 224)
