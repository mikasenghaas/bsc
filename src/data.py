# data.py
#  by: mika senghaas

import glob
import random
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from config import *
from utils import *

# load data
class ImageDataset(Dataset):
    def __init__(self, filepath: str, split = "train"):
        self.filepath = filepath
        self.split = split

        # image paths
        self.image_paths = load_labelled_image_paths(filepath)
        self.total_samples = len(self.image_paths)
        random.shuffle(self.image_paths)

        # transforms
        self.normalise = transforms.Normalize(MEAN, STD) # pyright:ignore

        # labels
        self.labels = load_labels(PROCESSED_DATA_PATH)
        self.num_classes = len(self.labels)
        self.label2id = { l: i for i, l in enumerate(self.labels) }
        self.id2label = { i: l for i, l in enumerate(self.labels) }

        # meta information to log
        self.meta = {
                'Labels': self.labels,
                'Number of Samples': self.total_samples,
                'Number of Classes': self.num_classes,
                'Splits': [TRAIN_RATIO, VAL_RATIO, TEST_RATIO]
                }

    def __getitem__(self, idx): # [x1, ..., x10]
        if self.split == 'val':
            idx += int(self.total_samples * TRAIN_RATIO)
        elif self.split == 'test':
            idx += int(self.total_samples * TRAIN_RATIO) + int(self.total_samples * VAL_RATIO)

        image_path, label = self.image_paths[idx]

        # load video to tensor
        image_tensor = torchvision.io.read_image(image_path) # C,H,W

        # get integer encoding of label
        label_id = self.label2id[label]

        return image_tensor, label_id

    def __len__(self):
        if self.split == "train":
            return int(self.total_samples * TRAIN_RATIO)
        elif self.split == "val":
            return int(self.total_samples * VAL_RATIO)
        elif self.split == "test":
            return int(self.total_samples * TEST_RATIO)
        else:
            raise Exception

class VideoDataset(Dataset):
    def __init__(self, filepath: str, split : str = "train"):
        self.filepath = filepath
        self.split = split

        # video paths
        self.video_paths = load_labeled_video_paths(filepath)
        self.num_samples = len(self.video_paths)

        # labels
        self.labels = load_labels(PROCESSED_DATA_PATH)
        self.num_classes = len(self.labels)
        self.label2id = { l: i for i, l in enumerate(self.labels) }
        self.id2label = { i: l for i, l in enumerate(self.labels) }

    def __getitem__(self, idx):
        video_path, label = self.video_paths[idx]

        # load video to tensor
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*"))) # assumes string sort matches frame sequence
        frames = []
        for frame_path in frame_paths:
            frame_tensor = torchvision.io.read_image(frame_path).float()
            frame_tensor = frame_tensor / 255.0
            frame_tensor = normalise_image(frame_tensor)

            frames.append(frame_tensor.unsqueeze(0))
        
        # concatenate to video tensor
        video_tensor = torch.cat(frames) # F, C, H, W

        # get integer encoding of label
        label_id = self.label2id[label]

        return video_tensor, label_id

    def __len__(self):
        if self.split == "train":
            return int(self.num_samples * TRAIN_RATIO)
        elif self.split == "val":
            return int(self.num_samples * VAL_RATIO)
        elif self.split == "test":
            return int(self.num_samples * TEST_RATIO)
        else:
            raise Exception
