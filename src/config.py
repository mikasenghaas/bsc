# config.py
#  by: mika senghaas

import os
import torch
from typing import NewType

# PATHS
RAW_DATA_PATH : str = os.path.join("src", "data", "raw")
PROCESSED_DATA_PATH : str = os.path.join("src", "data", "processed")
MODEL_PATH : str = os.path.join("src", "models")

# PREPROCESS
FPS : int = 2 # frames per second to sample

HEIGHT :int = 224 # image height
WIDTH : int = 224 # image width
CHANNELS : int = 3 # num channels (RGB)

# MODEL
CHECKPOINT = "MCG-NJU/videomae-base"
NUM_FRAMES = 16 # number of frames to sample

# OPTIMISER
LR : float = 1e-3

# TRAINING
DEVICE : str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

MAX_EPOCHS : int = 5
EVAL_ITERS : int = 100
BATCH_SIZE : int = 2

# TYPES
Annotation = NewType("AnnotationType", list[tuple[str, str]])
