"""
Module containing all the (default) configuration parameters for the project.

Author: Mika Senghaas
"""

import os
import torch

# GENERAL
SEED: int = 1
DEVICE: str = "cuda" if torch.cuda.is_available() \
    else "mps" if torch.backends.mps.is_available() else "cpu"

# PATHS
BASEPATH: str = "/".join(os.path.abspath(__file__).split("/")[:-2])
RAW_DATA_PATH: str = os.path.join(BASEPATH, "src", "data", "raw")
PROCESSED_DATA_PATH: str = os.path.join(BASEPATH, "src", "data", "processed")
MODEL_PATH: str = os.path.join(BASEPATH, "src", "models")

# CLASSES
CLASSES: list[str] = sorted([
    "Ground_Floor_Atrium",
    "Ground_Floor_Entrance_Yellow",
    "Ground_Floor_Entrance_Magenta",
    "Ground_Floor_Yellow_Area",
    "Ground_Floor_Magenta_Area",
    "Ground_Floor_Green_Area",
    "Ground_Floor_Red_Area",
    "Ground_Floor_Corridor_1",
    "Ground_Floor_Corridor_2",
    "Stairs_Atrium",
    "First_Floor_Mezzanine",
    "First_Floor_Yellow_Area",
    "First_Floor_Magenta_Area",
    "First_Floor_Green_Area",
    "First_Floor_Red_Area",
    "First_Floor_Corridor_1",
    "First_Floor_Corridor_2",
    "First_Floor_Library_1",
    "First_Floor_Library_2",
    "First_Floor_Library_3",
])
GROUND_FLOOR: list[str] = [x for x in CLASSES if x.find('Ground_Floor') >= 0]
FIRST_FLOOR: list[str] = [x for x in CLASSES if x.find(
    'First_Floor') >= 0 or x == "Stairs_Atrium"]

# PREPROCESS
MAX_LENGTH: int = 10
FPS: int = 4
HEIGHT: int = 224
WIDTH: int = 224

# DATA
SPLITS: list[str] = ["train", "val", "test"]
SPLIT: str = "train"
RATIO: float = 1.0
TRAIN_RATIO: float = 0.8
VAL_RATIO: float = 0.2

# TRANSFORMS
MEAN: torch.Tensor = torch.tensor([0.485, 0.456, 0.406])
STD: torch.Tensor = torch.tensor([0.229, 0.224, 0.225])

# TRAINING
PRETRAINED: bool = True
MAX_EPOCHS: int = 5
BATCH_SIZE: int = 32
LR: float = 1e-4
STEP_SIZE: int = 5
GAMMA: float = 0.1
LOG: bool = False

# EVALUATE
DURATION: int = 10
