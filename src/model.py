# model.py
#  by: mika senghaas

import torch
import torch.nn as nn
from torchvision.transforms.functional import normalize
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from transformers import logging

from pytorchvideo.transforms import Normalize

from torchvision.transforms import (
     Compose,
     Lambda,
)

from config import *
from utils import load_labels

logging.set_verbosity_error()

class VideoClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # loading labels
        self.labels = load_labels(PROCESSED_DATA_PATH)
        self.label2id = { l: i for i, l in enumerate(self.labels) }
        self.id2label = { i: l for i, l in enumerate(self.labels) }

        image_processor = VideoMAEImageProcessor()
        mean = image_processor.image_mean
        std = image_processor.image_std

        # processing
        self.transform=Compose([
                     Lambda(lambda x: x / 255.0),
                     Normalize(mean, std)
                     ])

        self.processor = VideoMAEImageProcessor.from_pretrained(CHECKPOINT,
                do_resize=False)

        # classifier
        self.videomae = VideoMAEForVideoClassification.from_pretrained(
                CHECKPOINT, 
                label2id=self.label2id, 
                id2label=self.id2label, 
                ignore_mismatched_sizes=True)

    def forward(self, x):
        """
        Forward pass to classify a video sequence to a location label.

        video : ArrayLike[B, T, C, H, W]
        """
        # process badge of images
        B,T,C,H,W = x.shape
        inputs = torch.empty(B, T, C, H, W)
        for i in range(B):
            processed = self.processor(list(x[i]), return_tensors="pt")['pixel_values']
            inputs[i] = processed.squeeze()

        outputs = self.videomae(inputs)
        logits = outputs.logits

        return logits

    def predict(self, x):
        pass
