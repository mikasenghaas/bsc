# utils.py
#  by: mika senghaas

from multiprocessing import cpu_count
import os
import timeit
import datetime

import ffmpeg
from moviepy.video.io.VideoFileClip import VideoFileClip
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

    return targets 

def load_video(src: str):
    return VideoFileClip(src, audio=False, target_resolution=(1920, 1080))

def clip_video(video: VideoFileClip, dst: str, start_time: int, end_time: int) -> None :
    assert end_time > start_time, "End Time must be larger thane Start Time"
    if mkdir(dst):
        clip = video.subclip(start_time, end_time)
        clip.write_videofile(dst, 
                audio=False, 
                logger=None,
                codec="libx264", 
                threads=cpu_count())

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
