# utils.py
#  by: mika senghaas

import argparse
import datetime
import glob
import math
import os
import timeit

import ffmpeg
from matplotlib import pyplot as plt
from matplotlib import animation
from termcolor import colored
import torch
from torchvision import transforms

from config import *
from model import MODELS
from utils import *

def load_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", "--model", type=str, choices=MODELS.keys(), help="Choose Model to train", required=True)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="Choose Maximum Epochs for Training")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Choose Batch Size for Training")
    parser.add_argument("--lr", type=float, default=LR, help="Choose Learning Rate For Training")
    parser.add_argument("--step-size", type=int, default=STEP_SIZE, help="Choose Step Size for Training")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Choose Gamma in LR Scheduler for Training")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=SAVE, help="Whether to save the Model after Training")

    return parser

def load_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", "--model", type=str, choices=MODELS.keys(), help="Choose Model to train", required=True)
    parser.add_argument("--filepath", type=str, default="", help="Filepath to model. If note chosen, will use most recent")

    return parser

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
    fig, ax = plt.subplots(figsize=(8,8), ncols=dim, nrows=dim) # pyright: ignore
    if not titles:
        titles = ["Unnamed Video Frame"] * n
    for i in range(dim):
        for j in range(dim):
            idx = i*(dim) + j
            show_image(image_tensors[idx], title=titles[idx], unnormalise=unnormalise, ax=ax[i, j]) # pyright: ignore
    plt.show()

def show_video(video_tensor : torch.Tensor, title: str | None = None, unnormalise : bool = True):
    assert video_tensor.ndim == 4, "Number of dimension must be 4"
    assert video_tensor.shape[1] == 3 , "Expects tensor of shape [T, C, H, W]"

    if unnormalise:
        video_tensor = torch.cat([unnormalise_image(frame).unsqueeze(0) for frame in video_tensor])

    F, C, H, W = video_tensor.shape
    # video_tensor = video_tensor.view(F, H, W, C)

    # Display the gif using matplotlib's animation module
    fig, ax = plt.subplots()
    ax.set_title(title) # type: ignore
    ax.set_xticks([]) # type: ignore
    ax.set_yticks([]) # type: ignore

    im = ax.imshow(transforms.ToPILImage()(video_tensor[0])) # type: ignore

    def animate(i):
        im.set_array(transforms.ToPILImage()(video_tensor[i]))
        return [im]

    a = animation.FuncAnimation(fig, animate, frames=len(video_tensor), interval=500, blit=True)
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
    labelled_image_paths = []

    for label_paths in glob.glob(os.path.join(filepath, "*")):
        for path in glob.glob(os.path.join(label_paths, "**/**")):
            labelled_image_paths.append((path, label_paths.split('/')[-1]))

    return labelled_image_paths

def load_labeled_video_paths(filepath: str):
    labelled_video_paths = []

    for label_paths in glob.glob(os.path.join(filepath, "*")):
        for path in glob.glob(os.path.join(label_paths, "*")):
            labelled_video_paths.append((path, label_paths.split('/')[-1]))
  
    return labelled_video_paths
