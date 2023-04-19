# data.py
#  by: mika senghaas

import os
import abc
import glob
import random
from collections import defaultdict

import torch
import torchvision
from torch.utils.data import Dataset

from config import (
    PROCESSED_DATA_PATH,
    DEFAULT_SPLIT,
    CLASSES,
    SPLITS,
    RATIO,
)
from utils import ls, load_labels


class BaseDataset(abc.ABC, Dataset):
    @staticmethod
    def default_config():
        """
        Staticmethod that contains all expected arguments for initialising a
        ImageDataset or VideoDataset object. The values are taken from config.py.
        """
        # default configuration for ImageDataset
        split: str = DEFAULT_SPLIT
        include_classes: list[str] = sorted(CLASSES)
        ratio: float = RATIO

        return {
            "split": split,
            "include_classes": include_classes,
            "ratio": ratio,
        }

    def __init__(self, **kwargs) -> None:
        super().__init__()
        # check all preconditions
        self.check_preconditions(kwargs)

        # save kwargs into variables
        self.split = kwargs["split"]
        self.ratio = kwargs["ratio"]
        self.include_classes = sorted(kwargs["include_classes"])

        # build class2id and id2class given sorted included classes
        self.class2id = {x: i for i, x in enumerate(self.include_classes)}
        self.id2class = {i: x for i, x in enumerate(self.include_classes)}

        # build dict of image paths and labels by video
        self.frames_by_clip: dict[str, list[tuple[str, str]]] = {}

        data_path = os.path.join(PROCESSED_DATA_PATH, self.split)
        for video_id in ls(data_path):
            # read frame paths into list
            all_paths = sorted(glob.glob(os.path.join(data_path, video_id, "*")))
            frame_paths = [path for path in all_paths if path.endswith(".jpg")]

            # read labels for frames into list
            labels_path = os.path.join(data_path, video_id, "labels.txt")
            labels = load_labels(labels_path)

            # zip frame paths and labels into list of tuples
            frames_with_labels = list(zip(frame_paths, labels))

            # add to dictionary
            self.frames_by_clip[video_id] = frames_with_labels

    def check_preconditions(self, kwargs):
        """
        Check if all preconditions are met before initialising the dataset.
        """
        # assert that kwargs contains all keys from default_config
        error_msg = (
            f"Class needs to be initialised with default values "
            f"for {self.default_config().keys()}"
        )
        assert len(kwargs.keys()) > 0 and all(
            x in kwargs.keys() for x in ImageDataset.default_config().keys()
        ), error_msg

        # assert that all included classes are valid (i.e. in CLASSES)
        error_msg = f"Only include classes from {CLASSES}"
        assert all(x in CLASSES for x in kwargs["include_classes"]), error_msg

        # assert that ratio is in ]0,1]
        error_msg = "Ratio needs to be in ]0,1]"
        assert 0.0 < kwargs["ratio"] <= 1.0, error_msg

        # assert that split is valid, i.e. in SPLITS
        error_msg = f"Split must be in {SPLITS}"
        assert kwargs["split"] in SPLITS, error_msg


class ImageDataset(BaseDataset):
    """
    ImageDataset extends the default torch.utils.Dataset class to sample random frames
    from an image dataset. Upon instantiation it loads all image paths to memory given
    a filepath to a directory, with the following expected structure:

    __filepath:
      |__ class1
          |__ frame1.jpg
          |__ ...
      |__ ...

    Parameters:

    split : str { train, val, test } = Samples from train, validation or test split
    include_classes : list[str] = List of classes to include
    ratio : float = Randomly sample ratio samples within each class
    """

    def __init__(self, **kwargs):
        # initialise base class to run preconditions
        super().__init__(**kwargs)

        # build dict of image paths and labels by class
        # subset on include_classes and randomly sample ratio
        self.frames_by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for video_id, frames_with_labels in self.frames_by_clip.items():
            for frame_path, label in frames_with_labels:
                if label in self.include_classes:
                    n = int(len(frames_with_labels) * self.ratio)
                    self.frames_by_class[label] = random.sample(frames_with_labels, n)

        # flatten dict into list of tuples
        self.data: list[tuple[str, str]] = []
        for frames_with_labels in self.frames_by_clip.values():
            frames, labels = zip(*frames_with_labels)  # unzip
            for frame, label in zip(frames, labels):
                self.data.append((frame, label))

        # shuffle list
        # random.Random(1).shuffle(self.data)

        # meta
        self.class_distribution = {k: len(v) for k, v in self.frames_by_class.items()}
        self.classes = list(self.class_distribution.keys())
        self.num_classes = len(self.classes)
        self.num_samples = len(self.data)

        # meta information to log
        self.meta = kwargs
        self.meta.update(
            {
                "class_distribution": self.class_distribution,
                "num_samples": self.num_samples,
                "num_classes": self.num_classes,
            }
        )

    def __getitem__(self, idx):  # [x1, ..., x10]
        # get image path and label
        image_path, label = self.data[idx]

        # load video to tensor
        image_tensor = torchvision.io.read_image(image_path)  # C,H,W

        # get integer encoding of label
        class_id = self.class2id[label]

        return image_tensor, class_id

    def __len__(self):
        return self.num_samples


class VideoDataset(BaseDataset):
    """
    VideoDataset extends the default torch.utils.Dataset class to sample random
    clips of a specified maximum number of frames from a video dataset. Upon
    instantiation it loads all video paths to memory given a filepath to a
    directory, with the following expected structure:

    __filepath:
      |__ clip1
      |   |__ video1
      |   |  |__ frame1.jpg
      |   |  |__ frame2.jpg
      |   |  |__ ...
      |   |__ ...
      |__ ...

    Parameters:

    split : str { train, val, test } = Samples from train, validation or test split
    include_classes : list[str] = List of classes to include
    ratio : float = Randomly sample ratio samples within each class

    TODO: Currently disregards include_classes and ratio (not well defined for video)
    """

    def __init__(self, **kwargs):
        # initialise base class to run preconditions
        super().__init__(**kwargs)

        # flatten dictionary
        self.data = [
            (video_id, frames_with_labels)
            for video_id, frames_with_labels in self.frames_by_clip.items()
        ]

        def partition(lst, n):
            """Yield successive n-sized chunks from lst."""
            res = []
            for i in range(0, len(lst), n):
                res.append(lst[i : i + n])
            return res[:-1]

        # partition frames in video in clips
        self.partioned_data = []
        for video_id, frames_with_labels in self.data:
            partioned_frames_with_labels = partition(frames_with_labels, 10)
            for i, frames_with_labels in enumerate(partioned_frames_with_labels):
                self.partioned_data.append(frames_with_labels)
        self.data = self.partioned_data

        # TODO: subset on include_classses?
        # TODO: randomly sample ratio?

        # meta
        self.num_samples = len(self.data)

        # store meta information
        self.meta = kwargs
        self.meta.update(
            {
                "num_samples": self.num_samples,
            }
        )

    def __getitem__(self, idx):
        # get video id and frames with labels
        frames_with_labels = self.data[idx]

        # unzip frames and labels
        frame_paths, labels = zip(*frames_with_labels)

        # load frame paths as list of image tenors
        frame_tensors = [torchvision.io.read_image(fp) for fp in frame_paths]

        # load labels and integer encoding
        class_ids = [self.class2id[label] for label in labels]

        # convert to tensors
        frame_tensors = torch.stack(frame_tensors)
        class_ids = torch.tensor(class_ids)

        return frame_tensors, class_ids

    def __len__(self):
        return self.num_samples

