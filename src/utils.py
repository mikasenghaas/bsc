# utils.py
#  by: mika senghaas

import argparse
import datetime
import glob
import math
import os
import pickle
import json
import timeit

import ffmpeg
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
from termcolor import colored
import torch
from torch.nn.functional import softmax
from torchvision import transforms

from config import *
from utils import *

def add_general_args(group):
    group.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=DEVICE, help="Training Device")

def add_wandb_args(group):
    group.add_argument("--wandb-log", action=argparse.BooleanOptionalAction, default=LOG, help="Log to WANDB")
    group.add_argument("--wandb-name", type=str, default="", help="Experiment Group (WANDB)")
    group.add_argument("--wandb-group", type=str, default="", help="Experiment Group (WANDB)")
    group.add_argument("--wandb-tags", nargs="+", default=[], help="Experiment Tags (WANDB)")

def add_data_args(group):
    group.add_argument("--filepath", type=str, default=len(load_labels(PROCESSED_DATA_PATH)), help=f"List of classes to include in training")
    group.add_argument("--include-classes", nargs="+", default=[], help=f"List of classes to include in training") # TODO
    group.add_argument("--all-classes", action=argparse.BooleanOptionalAction, default=False, help="Adds all classes in category 'Ground Floor' to training")
    group.add_argument("--ground-floor", action=argparse.BooleanOptionalAction, default=False, help="Adds all classes in category 'Ground Floor' to training")
    group.add_argument("--first-floor", action=argparse.BooleanOptionalAction, default=False, help="Adds all classes in category 'First Floor' to training")
    group.add_argument("--ratio", type=float, default=RATIO, help="Randomly sample a ratio of samples in every class")

def add_model_args(group):
    group.add_argument("-M", "--model", type=str, help="Model Identifier", required=True)
    group.add_argument("-V", "--version", type=str, default="latest", help="Model Version. Either 'latest' or 'vX'")
    group.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=PRETRAINED, help="Finetune pre-trained model")

def load_preprocess_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # preprocess args
    parser.add_argument("--split", choices=["train", "test"], default="train", help="Which split to extract clips from")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Maximum number of frames per extracted clip")
    parser.add_argument("--fps", type=int, default=FPS, help="Number of frames per second (FPS) to extract")

    args = parser.parse_args()
    return args

def load_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # data and model args
    general_group = parser.add_argument_group(title="General Arguments")
    model_group = parser.add_argument_group(title="Model Arguments")
    data_group = parser.add_argument_group(title="Data Arguments")
    wandb_group = parser.add_argument_group(title="W&B Arguments")

    add_model_args(model_group)
    add_data_args(data_group)
    add_general_args(general_group)
    add_wandb_args(wandb_group)

    # args only for training
    train_group = parser.add_argument_group(title="Training Arguments")
    train_group.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="Maximum Epochs")
    train_group.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch Size in Training and Validation Loader")
    train_group.add_argument("--lr", type=float, default=LR, help="Learning Rate for Optimiser")
    train_group.add_argument("--step-size", type=int, default=STEP_SIZE, help="Step Size for Scheduler")
    train_group.add_argument("--gamma", type=float, default=GAMMA, help="Gamma for Scheduler")

    # parse args
    args = parser.parse_args()

    include_classes = set(args.include_classes)
    if args.all_classes:
        include_classes |= set(CLASSES)
        args.ground_floor = True
        args.first_floor = True
    else:
        if args.ground_floor:
            include_classes |= set(GROUND_FLOOR)
        if args.first_floor:
            include_classes |= set(FIRST_FLOOR)
    args.include_classes = sorted(include_classes)

    if not args.wandb_log:
        args.wandb_name = False
        args.wandb_group = False
        args.wandb_tags = []

    return args

def load_infer_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # model and general args
    model_group = parser.add_argument_group(title="Model Arguments")
    general_group = parser.add_argument_group(title="Data Arguments")

    add_model_args(model_group)
    add_general_args(general_group)

    # infer args
    infer_group = parser.add_argument_group(title="Inference Arguments")
    infer_group.add_argument("--split", choices=SPLITS, default="test", help="From where to get the clip from")
    infer_group.add_argument("--clip", type=str, default=None, help="Which clip to sample")

    # parse args
    args = parser.parse_args()

    if args.clip != None:
        assert args.clip in os.listdir(os.path.join(RAW_DATA_PATH, args.split)), f"{args.clip} must be in {os.path.join(RAW_DATA_PATH, args.split)}"

    return args

def mkdir(filepath: str) -> bool:
    if filepath.find('.') >= 0:
        filepath = '/'.join(filepath.split('/')[:-1])
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        return True
    return False

def start_task(task: str, get_timer: bool = False) -> float | None:
    print(colored(f"> {task}", "green"))
    return timeit.default_timer() if get_timer else None

def end_task(task: str, start_time : float | None = None) -> None:
    if start_time:
        print(colored(f"> Finished {task} in {datetime.timedelta(seconds=int(timeit.default_timer() - start_time))}", "green"))
    else:
        print(colored(f"> Finished {task}", "green"))

def load_metadata(filepath: str) -> dict:
    meta = ffmpeg.probe(filepath).get('format')
    return meta

def load_annotations(filepath : str):
    targets  = []
    with open(filepath, "r") as file:
        for line in file:
            timestamp, label = line.strip().split(',')
            targets.append((timestamp, label))

    return targets # type: ignore

def normalise_image(image_tensor : torch.Tensor) -> torch.Tensor:
    return (image_tensor - MEAN[:, None, None]) / STD[:, None, None]

def unnormalise_image(image_tensor : torch.Tensor) -> torch.Tensor:
    return (image_tensor * STD[:, None, None] + MEAN[:, None, None])

def get_summary(args: dict):
    return pd.Series(args)

def show_image(image_tensor : torch.Tensor, title : str | None= None, unnormalise : bool = False, ax : plt.Axes = None, show : bool = False):
    assert image_tensor.ndim == 3, "Number of dimension must be 3"
    assert image_tensor.shape[0] == 3 , "Expects tensor of shape [C, H, W]"

    if ax==None:
        _, ax = plt.subplots() # pyright: ignore
    if unnormalise: 
        image_tensor = unnormalise_image(image_tensor)
    image_tensor = transforms.ToPILImage()(image_tensor.to('cpu'))
    ax.imshow(image_tensor)
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if title != None:
        ax.set_title(title)
    if show:
        plt.show()

def show_images(image_tensors : torch.Tensor, titles : list[str] | None= None, unnormalise : bool = False, ax : plt.Axes = None, show : bool = False):
    assert image_tensors.ndim == 4, "Number of dimension must be 3"
    assert image_tensors.shape[1] == 3 , "Expects tensor of shape [F, C, H, W]"

    n = len(image_tensors)
    dim = int(math.sqrt(n))
    fig, ax = plt.subplots(figsize=(16,16), ncols=dim, nrows=dim) # pyright: ignore
    if titles == None:
        titles = ["Unnamed Video Frame"] * n
    for i in range(dim):
        for j in range(dim):
            idx = i*(dim) + j
            show_image(image_tensors[idx], title=titles[idx], unnormalise=unnormalise, ax=ax[i, j]) # pyright: ignore
    plt.show()

def show_video(video_tensor : torch.Tensor, title: str | None = None, unnormalise : bool = True, interval : int = 500):
    assert video_tensor.ndim == 4, "Number of dimension must be 4"
    assert video_tensor.shape[1] == 3 , "Expects tensor of shape [T, C, H, W]"

    if unnormalise:
        video_tensor = torch.cat([unnormalise_image(frame).unsqueeze(0) for frame in video_tensor])

    # Display the gif using matplotlib's animation module
    fig, ax = plt.subplots()
    ax.set_title(title) # type: ignore
    ax.set_xticks([]) # type: ignore
    ax.set_yticks([]) # type: ignore

    im = ax.imshow(transforms.ToPILImage()(video_tensor[0])) # type: ignore

    def animate(i):
        im.set_array(transforms.ToPILImage()(video_tensor[i]))
        return [im]

    a = animation.FuncAnimation(fig, animate, frames=len(video_tensor), interval=interval, blit=True)
    plt.show()

def timestamp_to_second(timestamp : str) -> int:
  mm, ss = map(int, timestamp.split(':'))
  return mm * 60 + ss

def get_label(second_in_video: int, annotations: list[tuple[str, str]]) -> str:
  # assumes targets is sorted by timestamp
  prev_label = annotations[0][1]
  for i in range(1, len(annotations)):
    timestamp, label = annotations[i]
    seconds = timestamp_to_second(timestamp)
    if second_in_video < seconds:
      return prev_label
    prev_label = label

  return annotations[-1][1]

def load_labels(filepath: str) -> list[str]:
    return sorted(os.listdir(filepath))

def load_labelled_image_paths(filepath: str):
    labelled_image_paths = {}

    for label_paths in glob.glob(os.path.join(filepath, "*")):
        label = label_paths.split('/')[-1]
        labelled_image_paths[label] = []
        for path in glob.glob(os.path.join(label_paths, "**/**")):
            labelled_image_paths[label].append((path, label))

    return labelled_image_paths

def load_labeled_video_paths(filepath: str):
    labelled_video_paths = []

    for label_paths in glob.glob(os.path.join(filepath, "*")):
        for path in glob.glob(os.path.join(label_paths, "*")):
            labelled_video_paths.append((path, label_paths.split('/')[-1]))
  
    return labelled_video_paths

def save_pickle(obj, filepath: str):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def save_json(obj, filepath: str):
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=4, sort_keys=True)

def load_json(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def get_predictions(model, transforms, loader):
    # load and predict on test split
    model.to(DEVICE)
    y_true, y_pred, y_probs = None, None, None
    for batch_num, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # predict test samples
        logits = model(transforms(inputs))
        probs = softmax(logits, 1).detach() # B, 7
        preds = logits.argmax(-1)

        if batch_num == 0:
            y_true = labels.cpu()
            y_pred = preds.cpu()
            y_probs = probs.cpu()
        else:
            y_true = np.concatenate((y_true, labels.cpu()))
            y_pred = np.concatenate((y_pred, preds.cpu()))
            y_probs = np.concatenate((y_probs, probs.cpu()))

    return y_true, y_pred, y_probs
