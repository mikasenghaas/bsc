# data.py
#  by: mika senghaas

from torch.utils.data import Dataset
from torchvision.io import read_video

from config import *
from utils import *

class TrainData(Dataset):
    def __init__(self, filepath: str, model):
        self.filepath = filepath
        processor = model.processor

        # get transformations of image processor from model
        self.mean, self.std = processor.image_mean, processor.image_std
        height = width = processor.size["shortest_edge"]
        self.resize_to = (height, width)

        # video paths
        self.video_paths = load_labeled_video_paths(filepath)
        self.num_videos = len(self.video_paths)

        # labels
        self.labels = load_labels(PROCESSED_DATA_PATH)
        self.label2id = { l: i for i, l in enumerate(self.labels) }
        self.id2label = { i: l for i, l in enumerate(self.labels) }

    def __getitem__(self, idx):
        video_path, label = self.video_paths[idx]

        # load video to tensor
        video = read_video(video_path, pts_unit="sec")
        video_tensor, _, _ = video # (f, h, w, c)

        # reorder
        F,H,W,C = video_tensor.shape
        video_tensor = video_tensor.view(F, C, H, W)

        # get integer encoding of label
        label_id = self.label2id[label]

        return video_tensor, label_id

    def __len__(self):
        return self.num_videos
