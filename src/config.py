# config.py
#  by: mika senghaas

import torch
from typing import NewType

# PATHS
RAW_DATA_PATH : str = "src/data/raw"
PROCESSED_DATA_PATH : str = "src/data/processed"

# PREPROCESS
CLIP_LENGTH : int = 10 # video length
FPS : int = 4 # downsample to 25% of frames
H :int = 480 # image height
W : int = 270 # image width
C : int = 3 # num channels (RGB)

# MODEL

# OPTIMISER
LR : float = 1e-3

# TRAINING
DEVICE : str = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_ITERS : int = 10 ** 5
EVAL_ITERS : int = 100
BATCH_SIZE : int = 16

# TYPES
Annotation = NewType("AnnotationType", list[tuple[str, str]])
