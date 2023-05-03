from data import VideoDataset, ImageDataset
from transform import VideoTransform, ImageTransform
from model import VideoClassifier, ImageClassifier

MODULES = {
    "video": {
        "data": VideoDataset,
        "transform": VideoTransform,
        "model": VideoClassifier,
    },
    "image": {
        "data": ImageDataset,
        "transform": ImageTransform,
        "model": ImageClassifier,
    },
}
