# data.py
#  by: mika senghaas

import os
import glob

import torch
import pytorchvideo

from pytorchvideo.data import LabeledVideoDataset
from torchvision.datasets import ImageFolder

from config import (
    IMAGE_DATA_PATH,
    VIDEO_DATA_PATH,
)


class ImageDataset:
    def __init__(self, split, transform):
        path = os.path.join(IMAGE_DATA_PATH, split)
        self.dataset = ImageFolder(root=path, transform=transform)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)


class VideoDataset:
    """
    VideoDataset is a wrapper for `pytorchvideo.data.LabelledDataset` that samples
    random clips from a video dataset. Upon instantiation it loads all video paths to
    memory and builds a dataset that can easily be sampled from.


    Parameters:
        split : str { train, test } = Samples from train, validation or test split

    """

    def __init__(self, split, clip_duration, transform):
        self.split = split
        self.transform = split

        # get all paths to videos
        video_paths = glob.glob(
            os.path.join(VIDEO_DATA_PATH, self.split, "**/*.mp4"), recursive=True
        )

        # build list of tuples (video_path, {label: label})
        labeled_video_paths = [
            (path, {"label": path.split("/")[-2]}) for path in video_paths
        ]

        self.dataset = LabeledVideoDataset(
            labeled_video_paths=labeled_video_paths,
            clip_sampler=pytorchvideo.data.UniformClipSampler(
                clip_duration=clip_duration
            ),
            video_sampler=torch.utils.data.RandomSampler,
            transform=transform,
            decode_audio=False,
        )

        # compute class-id mapping
        self.classes = sorted(
            list(set([label_dict["label"] for _, label_dict in labeled_video_paths]))
        )
        self.class2id = {label: i for i, label in enumerate(self.classes)}
        self.id2class = {i: label for i, label in enumerate(self.classes)}

    def __getitem__(self, _):
        # get next batch
        batched_video = self.dataset.__next__()
        video = batched_video["video"]
        label = batched_video["label"]

        # integer encode label label
        encoded_class = torch.tensor(self.class2id[label], dtype=torch.long)

        return video, encoded_class

    def __len__(self):
        return len(self.dataset.video_sampler)

    def __iter__(self):
        return self.dataset.__iter__()
