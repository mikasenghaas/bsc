# data.py
#  by: mika senghaas

import glob
import random
import torchvision
from torch.utils.data import Dataset

from config import *
from utils import *

# random.seed(SEED)

# load data
class ImageDataset(Dataset):
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
    @staticmethod
    def default_config():
        """
        Staticmethod that contains all expected arguments for initialising a ImageDataset 
        object. The values are taken from config.py. Use this method either on the the class like
        ImageDataset.default_config()
        """
        # default configuration for ImageDataset
        split : str = SPLIT
        include_classes : list[str] = sorted(CLASSES)
        ratio : float = RATIO

        return {"split": split,
                "include_classes": include_classes,
                "ratio": ratio, }

    def __init__(self, **kwargs):
        assert len(kwargs.keys()) > 0 and all(x in kwargs.keys() for x in ImageDataset.default_config().keys()), f"Class needs to be initialised with default values for 'filepath', 'split', 'include_classes' and 'ratio'."

        # save kwargs into variables
        self.split = kwargs['split']
        self.include_classes = kwargs['include_classes']
        self.ratio = kwargs['ratio']

        # pre conditions
        assert self.split in SPLITS, "Split must be either 'train', 'val' or 'test'"
        assert all(x in CLASSES for x in self.include_classes), f"Only include classes from {CLASSES}"
        assert self.ratio > 0.0 and self.ratio <= 1.0, "Ratio needs to be in ]0,1]"

        # get all image paths by class
        self.images_by_class : dict[str, list[str]] = load_labelled_image_paths(PROCESSED_DATA_PATH)

        # subset to only include specified classes
        self.images_by_class = {k: self.images_by_class[k] for k in self.include_classes if k in self.images_by_class}
        assert len(self.images_by_class) == len(self.include_classes)

        # randomly sample num_classes
        if self.ratio != 1.0:
            for key, val in self.images_by_class.items():
                n = int(len(val) * self.ratio)
                self.images_by_class[key] = random.sample(val, n)

        # convert to shuffled flattened list of paths
        image_paths = [image_paths for image_paths in self.images_by_class.values()]
        self.image_paths = [item for sublist in image_paths for item in sublist] # flatten
        random.shuffle(self.image_paths)

        # meta
        self.class_distribution = { k: len(v) for k, v in self.images_by_class.items()}
        self.classes = list(self.class_distribution.keys())
        self.num_classes = len(self.classes)
        self.num_samples = len(self.image_paths)
        self.class2id = { l: i for i, l in enumerate(self.classes) }
        self.id2class = { i: l for i, l in enumerate(self.classes) }

        # meta information to log
        self.meta = kwargs
        self.meta.update({
                'class_distribution': self.class_distribution,
                'number_of_samples': self.num_samples,
                'number_of_classes': self.num_classes,
                })

    def __getitem__(self, idx): # [x1, ..., x10]
        if self.split == 'val':
            idx += int(self.num_samples * TRAIN_RATIO)
        elif self.split == 'test':
            idx += int(self.num_samples * TRAIN_RATIO) + int(self.num_samples * VAL_RATIO)

        image_path, label = self.image_paths[idx]

        # load video to tensor
        image_tensor = torchvision.io.read_image(image_path) # C,H,W

        # get integer encoding of label
        class_id = self.class2id[label]

        return image_tensor, class_id

    def __len__(self):
        if self.split == "train":
            return int(self.num_samples * TRAIN_RATIO)
        elif self.split == "val":
            return int(self.num_samples * VAL_RATIO)
        elif self.split == "test":
            return int(self.num_samples * TEST_RATIO)
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
