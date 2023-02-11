# profile.py
#  by: mika senghaas

from timeit import default_timer

import numpy as np
from torch.utils.data import DataLoader

from config import *
from utils import *
from data import TrainData
from model import VideoClassifier

def repeat(fun, *args, name: str="Anonymous", n: int=3):
    times = np.zeros(n)
    for i in range(n):
        start_time = default_timer()
        fun(*args)
        end_time = default_timer() - start_time

        times[i] = end_time

    print(f"Ran {name} {n} times: {round(np.mean(times), 2)}s (SD: {round(np.std(times),2)})")

def load_sample(data):
    video, label = next(iter(data))
    return video, label

def load_batch(loader):
    videos, labels = next(iter(loader))
    return videos, labels

def main():
    model = VideoClassifier()
    data = TrainData(PROCESSED_DATA_PATH, model)
    loader = DataLoader(data, batch_size=8)

    start_task("Checking load_sample()")
    repeat(load_sample, data, name="load_sample()")

    start_task("Checking load_batch()")
    repeat(load_batch, loader, name="load_batch()")

if __name__ == "__main__":
    main()
