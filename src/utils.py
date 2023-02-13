# utils.py
#  by: mika senghaas

import datetime
import glob
import os
import timeit

import ffmpeg
from matplotlib import pyplot as plt
from matplotlib import animation
from termcolor import colored

from config import *

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

def load_annotations(filepath : str) -> Annotation:
    targets  = []
    with open(filepath, "r") as file:
        for line in file:
            timestamp, label = line.strip().split(',')
            targets.append((timestamp, label))

    return targets # type: ignore

def display_video(video_tensor : torch.Tensor, title: str = "Example Sample") -> None:
    assert video_tensor.ndim == 4, "Number of dimension must be 4"
    assert video_tensor.shape[1] == 3 , "Expects tensor of shape [T, C, H, W]"

    B, C, H, W = video_tensor.shape
    video_tensor = video_tensor.view(B, H, W, C)

    # Display the gif using matplotlib's animation module
    fig, ax = plt.subplots()
    ax.set_title(title) # type: ignore
    ax.set_xticks([]) # type: ignore
    ax.set_yticks([]) # type: ignore

    im = ax.imshow(video_tensor[0]) # type: ignore

    def animate(i):
        im.set_array(video_tensor[i])
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

def load_labeled_video_paths(filepath: str):
    labeled_video_paths = []
    for filename in glob.iglob(filepath + "/**/*", recursive=True):
        if filename.find('.') >= 0:
            label = filename.split('/')[-2]
            labeled_video_paths.append((filename, label))
    return sorted(labeled_video_paths)
