# config.py
#  by: mika senghaas

import os
import torch

# GENERAL
SEED        : int                   = 1
DEVICE      : str                   = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# PATHS
BASEPATH : str                      = "/".join(os.path.abspath(__file__).split("/")[:-2])
RAW_DATA_PATH : str                 = os.path.join(BASEPATH, "src", "data", "raw")
PROCESSED_DATA_PATH : str           = os.path.join(BASEPATH, "src", "data", "processed")
MODEL_PATH  : str                   = os.path.join(BASEPATH, "src", "models")

# CLASSES
CLASSES : list[str]                 = [label for label in os.listdir(os.path.join(PROCESSED_DATA_PATH, "train")) if not label.startswith(".")]
GROUND_FLOOR : list[str]            = [x for x in CLASSES if x.find('Ground_Floor') >= 0]
FIRST_FLOOR : list[str]             = [x for x in CLASSES if x.find('First_Floor') >= 0 or x == "Stairs_Atrium"]

# PREPROCESS
MAX_LENGTH  : int                   = 10
FPS         : int                   = 4
HEIGHT      : int                   = 224
WIDTH       : int                   = 224

# DATA
SPLITS      : list[str]             = ["train", "val", "test"]
SPLIT       : str                   = "test"
RATIO       : float                 = 1.0
TRAIN_RATIO : float                 = 0.8
VAL_RATIO   : float                 = 0.2

# TRANSFORMS
MEAN        : torch.Tensor          = torch.tensor([0.485, 0.456, 0.406])
STD         : torch.Tensor          = torch.tensor([0.229, 0.224, 0.225])

# TRAINING
PRETRAINED  : bool                  = True
MAX_EPOCHS  : int                   = 5
BATCH_SIZE  : int                   = 32
LR          : float                 = 1e-4
STEP_SIZE   : int                   = 5
GAMMA       : float                 = 0.1
LOG         : bool                  = False

# EVALUATE
DURATION    : int                   = 10
