"""
Module for utility functions.

Author: Mika Senghaas
"""

import argparse
import datetime
import json
import math
import os
import pickle
import timeit
from typing import Any

import cv2
import pandas as pd
import torch
from matplotlib import animation
from matplotlib import pyplot as plt
from termcolor import colored
from torchvision import transforms

from config import (
    LOG,
    RAW_DATA_PATH,
    SPLITS,
    MEAN,
    STD,
)


def add_wandb_args(group: argparse._ArgumentGroup) -> None:
    """
    Add W&B arguments to argparse group.

    Args:
        group (argparse._ArgumentGroup): Argument group to add arguments to
    """
    group.add_argument(
        "--wandb-log",
        action=argparse.BooleanOptionalAction,
        default=LOG,
        help="Log to WANDB",
    )
    group.add_argument(
        "--wandb-name", type=str, default="", help="Experiment Group (WANDB)"
    )
    group.add_argument(
        "--wandb-group", type=str, default="", help="Experiment Group (WANDB)"
    )
    group.add_argument(
        "--wandb-tags", nargs="+", default=[], help="Experiment Tags (WANDB)"
    )


def add_model_args(group):
    """
    Add arguments for model to argparse group.

    Args:
        group (argparse._ArgumentGroup): Argument group to add arguments to
    """
    group.add_argument(
        "-M", "--model", type=str, help="Model Identifier", required=True
    )
    group.add_argument(
        "-V",
        "--version",
        type=str,
        default="latest",
        help="Model Version. Either 'latest' or 'vX'",
    )


def load_preprocess_args() -> argparse.Namespace:
    """
    Return parsed arguments for script preprocess.py.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()  # create parser

    # preprocess args
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Which split to extract clips from",
    )
    parser.add_argument(
        "--vid-length",
        type=int,
        default=5,
        help="Max. Length of video in seconds",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Size of crop for each frame/ clip",
    )

    args = parser.parse_args()
    return args


def load_train_args() -> argparse.Namespace:
    """
    Return parsed arguments for script train.py.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()  # create parser

    # create general, model, data and wandb args
    model_group = parser.add_argument_group(title="Model Arguments")
    wandb_group = parser.add_argument_group(title="W&B Arguments")

    # add args to each group
    add_model_args(model_group)
    add_wandb_args(wandb_group)

    # add training args
    train_group = parser.add_argument_group(title="Training Arguments")
    train_group.add_argument("--epochs", type=int, help="Maximum Epochs")
    train_group.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size in Training and Validation Loader",
    )
    train_group.add_argument("--lr", type=float, help="Learning Rate for Optimiser")

    # parse args
    args = parser.parse_args()

    if not args.wandb_log:
        args.wandb_name = False
        args.wandb_group = False
        args.wandb_tags = []

    return args


def load_eval_args() -> argparse.Namespace:
    """
    Return parsed arguments for script eval.py.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()

    # model and general args
    model_group = parser.add_argument_group(title="Model Arguments")

    add_model_args(model_group)

    # parse args
    args = parser.parse_args()

    return args


def load_infer_args() -> argparse.Namespace:
    """
    Return parsed arguments for script infer.py.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()

    # model and general args
    model_group = parser.add_argument_group(title="Model Arguments")

    add_model_args(model_group)

    # infer args
    infer_group = parser.add_argument_group(title="Inference Arguments")
    infer_group.add_argument(
        "--gradcam",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run and overlay GradCam during inference",
    )
    infer_group.add_argument(
        "--split",
        choices=SPLITS,
        default="test",
        help="From where to get the clip from",
    )
    infer_group.add_argument(
        "--clip", type=str, default=None, help="Which clip to sample"
    )

    # parse args
    args = parser.parse_args()

    if args.clip is not None:
        assert args.clip in os.listdir(
            os.path.join(RAW_DATA_PATH, args.split)
        ), f"{args.clip} must be in {os.path.join(RAW_DATA_PATH, args.split)}"

    return args


def mkdir(filepath: str) -> bool:
    """
    Create a directory filepath if it does not exist.

    Args:
        filepath (str): Path to directory

    Returns:
        bool: True if directory was created, False if it already exists
    """
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        return True
    return False


def ls(filepath: str) -> list[str]:
    """
    Return all file and directory names in sorted order within the given filepath
    (excluding hidden files).

    Args:
        filepath (str): Path to the directory containing the labelled images

    Returns:
        list: List of files and directories within the given filepath
    """
    return sorted([path for path in os.listdir(filepath) if not path.startswith(".")])


def start_task(task: str, get_timer: bool = False) -> float | None:
    """
    Print a colored task to the console; Optionally returns which can be passed to
    end_task, otherwise None.

    Args:
        task (str): Task to print
        get_timer (bool, optional): Whether to return a timer. Defaults to False.

    Returns:
        float | None: Timer if get_timer is True, otherwise None
    """
    print(colored(f"> {task}", "green"))
    return timeit.default_timer() if get_timer else None


def end_task(task: str, start_time: float | None = None) -> None:
    """
    Print colored task signalling that the task has finishedk; Optionally takes a timer
    generated by start_task to print the time taken for the task.

    Args:
        task (str): Task to print
        start_time (float | None, optional): Timer generated by start_task.

    Returns:
        None
    """
    if start_time:
        current_time = int(timeit.default_timer())
        time_delta = datetime.timedelta(seconds=current_time - start_time)
        print(colored(f"> Finished {task} in {time_delta}", "green"))
    else:
        print(colored(f"> Finished {task}", "green"))


def get_progress_bar(
    epoch: int,
    max_epochs: int,
    batch: int,
    training_time: float,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    train: bool,
):
    """
    Return a string representing the current training progress.

    Args:
        epoch (int): Current epoch
        max_epochs (int): Total number of epochs
        batch (int): Current batch
        running_training_time (float): Total time spent training
        running_inference_time (float): Total time spent inferring
        samples_seen (int): Total number of samples seen
        train_loss (float): Current training loss
        train_acc (float): Current training accuracy
        val_loss (float): Current validation loss
        val_acc (float): Current validation accuracy

    Returns:
        str: String representing the current training progress
    """
    # format function inputs
    a = f"{str(epoch)}".zfill(len(str(max_epochs)))
    b = f"{str(batch)}".zfill(3)
    c = f"{round(training_time * 1000, 2)}ms"
    d = f"{train_loss:.3f}"
    e = f"{(train_acc * 100):.1f}%"
    f = f"{val_loss:.3f}"
    g = f"{(val_acc * 100):.1f}%"

    return (
        f"{a}/{max_epochs} | {'Train' if train else 'Val'} | "
        f"{b} | {c} | "
        f"Train: {d} ({e}) | Val: {f} ({g})"
    )


def load_metadata(filepath: str) -> dict:
    """
    Load metadata from a video file.

    Args:
        filepath (str): Path to video file

    Returns:
        tuple[int, int]: Return a tuple of form (duration, frame_count)
    """
    video = cv2.VideoCapture(filepath)

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = int(frame_count // video.get(cv2.CAP_PROP_FPS))

    return duration, frame_count


def load_annotations(filepath: str) -> list[tuple[str, str]]:
    """
    Load annotations from a csv file. Goes through each line and splits it into a tuple
    of (timestamp, label) and appends to a list of tuples.

    Args:
        filepath (str): Path to csv file

    Returns:
        list[tuple[str, str]]: List of tuples of (timestamp, label)
    """
    targets = []
    with open(filepath, "r") as file:
        for line in file:
            timestamp, label = line.strip().split(",")
            targets.append((timestamp, label))

    return targets  # type: ignore


def normalise_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalise an image tensor with the mean and standard deviation of the ImageNet
    dataset.

    Args:
        image_tensor (torch.Tensor): Image tensor to normalise

    Returns:
        torch.Tensor: Normalised image tensor
    """
    return (image_tensor - MEAN[:, None, None]) / STD[:, None, None]


def unnormalise_image(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Unnormalise an image tensor with the mean and standard deviation of the
    ImageNet dataset.

    Args:
        image_tensor (torch.Tensor): Image tensor to unnormalise

    Returns:
        torch.Tensor: Unnormalised image tensor
    """
    return image_tensor * STD[:, None, None] + MEAN[:, None, None]


def get_summary(args: dict) -> pd.Series:
    """
    Return a summary of the arguments as a pandas Series given a dictionary
    of arguments.

    Args:
        args (dict): Dictionary of arguments

    Returns:
        pd.Series: Summary of arguments
    """
    return pd.Series(args)


def show_image(
    image_tensor: torch.Tensor,
    title: str | None = None,
    unnormalise: bool = False,
    ax: plt.Axes = None,
    show: bool = False,
) -> None:
    """
    Display an image tensor using matplotlib. Can take unnormalised image tensor with
    unnormalise=False or normalised image tensor with unnormalise=True.

    Optionally can be given a custom title. If ax is not None, the image will be
    displayed on the given axis, otherwise a new axis will be created.
    The image will be displayed if show=True, otherwise it will not be displayed.

    Args:
        image_tensor (torch.Tensor): Image tensor [C, H, W] to display
        title (str | None, optional): Title of the image. Defaults to None.
        unnormlaise (bool, optional): Whether the image tensor is normalised or not.
        ax (plt.Axes, optional): Axis to display the image on. Defaults to None.
        show (bool, optional): Whether to display the image. Defaults to False.

    Returns:
        None
    """
    assert image_tensor.ndim == 3, "Number of dimension must be 3"
    assert image_tensor.shape[0] == 3, "Expects tensor of shape [C, H, W]"

    if ax is None:
        _, ax = plt.subplots()  # pyright: ignore
    if unnormalise:
        image_tensor = unnormalise_image(image_tensor)
    image_tensor = transforms.ToPILImage()(image_tensor.to("cpu"))
    ax.imshow(image_tensor)
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if title is not None:
        ax.set_title(title)
    if show:
        plt.show()


def show_images(
    image_tensors: torch.Tensor,
    titles: list[str] | None = None,
    unnormalise: bool = False,
    ax: plt.Axes = None,
):
    """
    Display a grid of image tensors using matplotlib by utilising the show_image
    function.

    Can take unnormalised image tensor with unnormalise=False or normalised image
    tensor with unnormalise=True.
    Grid of images is shown by default.
    """
    assert image_tensors.ndim == 4, "Number of dimension must be 3"
    assert image_tensors.shape[1] == 3, "Expects tensor of shape [F, C, H, W]"

    n = len(image_tensors)
    dim = int(math.sqrt(n))
    fig, ax = plt.subplots(figsize=(16, 16), ncols=dim, nrows=dim)  # pyright: ignore
    if titles is None:
        titles = ["Unnamed Video Frame"] * n
    for i in range(dim):
        for j in range(dim):
            idx = i * (dim) + j
            show_image(
                image_tensors[idx],
                title=titles[idx],
                unnormalise=unnormalise,
                ax=ax[i, j],
            )  # pyright: ignore
    plt.show()

    return fig


def show_video(
    video_tensor: torch.Tensor,
    title: str | None = None,
    unnormalise: bool = True,
) -> None:
    """
    Display a video tensor using matplotlib's animation module.
    Can take unnormalised video tensor with unnormalise=False or
    normalised video tensor with unnormalise=True.

    Args:
        video_tensor (torch.Tensor): Video tensor [T, C, H, W] to display
        title (str | None, optional): Title of the video. Defaults to None.
        unnormalise (bool, optional): Whether the video tensor is normalised or not.

    Returns:
        None
    """
    assert video_tensor.ndim == 4, "Number of dimension must be 4"
    assert video_tensor.shape[1] == 3, "Expects tensor of shape [T, C, H, W]"

    if unnormalise:
        video_tensor = torch.cat(
            [unnormalise_image(frame).unsqueeze(0) for frame in video_tensor]
        )

    # Display the gif using matplotlib's animation module
    fig, ax = plt.subplots()
    ax.set_title(title)  # type: ignore
    ax.set_xticks([])  # type: ignore
    ax.set_yticks([])  # type: ignore

    im = ax.imshow(transforms.ToPILImage()(video_tensor[0]))  # type: ignore

    def animate(i):
        im.set_array(transforms.ToPILImage()(video_tensor[i]))
        return [im]

    _ = animation.FuncAnimation(
        fig, animate, frames=len(video_tensor), interval=1, blit=True
    )
    plt.show()


def timestamp_to_second(timestamp: str) -> int:
    """
    Convert a timestamp in the format mm:ss to seconds.

    Args:
        timestamp (str): Timestamp in the format mm:ss

    Returns:
        int: Timestamp in seconds
    """
    mm, ss = map(int, timestamp.split(":"))
    return mm * 60 + ss


def get_label(second_in_video: int, annotations: list[tuple[str, str]]) -> str:
    """
    Extract the location label of a video given the second in the video and the
    annotations. Assumes annotations are sorted by timestamp.

    Args:
        second_in_video (int): Second in the video
        annotations (list[tuple[str, str]]): List of tuples [(timestamp, label)]

    Returns:
        str: Location label
    """
    # set first label as default
    prev_label = annotations[0][1]
    # iterate over all other annotations (index position 1 to end)
    for i in range(1, len(annotations)):
        # extract timestamp and label from current annotation
        timestamp, label = annotations[i]

        # convert timestamp to seconds
        seconds = timestamp_to_second(timestamp)

        # if second in video is less than the current timestamp, return the
        # previous label
        if second_in_video < seconds:
            return prev_label

        # otherwise, set the current label as the previous label
        prev_label = label

    # if second in video is greater than all timestamps, return the last label
    return annotations[-1][1]


def load_labels(filepath: str) -> list[str]:
    """
    Load a list of labels from a text file. Assumes that each label
    is on a new line.

    Args:
        filepath (str): Path to the text file

    Returns:
        list[str]: List of labels
    """
    labels = []
    with open(filepath, "r") as f:
        for line in f:
            labels.append(line.strip())
    return labels


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save a pickle file to the given filepath.

    Args:
        obj (Any): Object to be pickled
        filepath (str): Path to save the pickle file

    Returns:
        None
    """
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Load a pickle file from the given filepath.

    Args:
        filepath (str): Path to load the pickle file

    Returns:
        Any: Object loaded from the pickle file
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, filepath: str) -> None:
    """
    Save a json file to the given filepath.

    Args:
        obj (Any): Object to be saved
        filepath (str): Path to save the json file

    Returns:
        None
    """
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=4, sort_keys=True)


def load_json(filepath: str) -> Any:
    """
    Load a json file from the given filepath.

    Args:
        filepath (str): Path to load the json file

    Returns:
        Any: Object loaded from the json file
    """
    with open(filepath, "r") as f:
        return json.load(f)
