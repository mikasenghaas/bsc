# config.py
#  by: mika senghaas

import os
import torch

# PATHS
RAW_DATA_PATH : str         = os.path.join("src", "data", "raw")
PROCESSED_DATA_PATH : str   = os.path.join("src", "data", "processed")
MODEL_PATH  : str           = os.path.join("src", "models")

# PREPROCESS
HEIGHT      : int           = 224 # image height
WIDTH       : int           = 224 # image width
MEAN        : torch.Tensor  = torch.tensor([0.485, 0.456, 0.406])
STD         : torch.Tensor  = torch.tensor([0.229, 0.224, 0.225])

# DATA
SPLITS      : list[str]     = ["train", "val", "test"]
TRAIN_RATIO : float         = 0.7
VAL_RATIO   : float         = 0.1
TEST_RATIO  : float         = 0.2

# TRAINING
DEVICE      : str           = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MAX_EPOCHS  : int           = 30
BATCH_SIZE  : int           = 16
LR          : float         = 1e-3
STEP_SIZE   : int           = 7
GAMMA       : float         = 0.1
SAVE        : bool          = False
FINETUNE    : bool          = True
