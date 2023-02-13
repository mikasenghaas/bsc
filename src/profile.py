# profile.py
#  by: mika senghaas

from timeit import default_timer
from termcolor import colored

import numpy as np
from torch.utils.data import DataLoader
from transformers import logging

from config import *
from utils import *
from data import TrainData
from model import VideoClassifier

logging.set_verbosity_error()

def repeat(fun, *args, name: str="Anonymous", n: int=3):
    times = np.zeros(n)
    for i in range(n):
        start_time = default_timer()
        fun(*args)
        end_time = default_timer() - start_time

        times[i] = end_time

    mean_time, sd_time = np.mean(times), np.std(times)
    print(f"Ran {name} {n} times: {round(mean_time, 2)}s (SD: {round(sd_time,2)})")

    return mean_time, sd_time

def load_sample(data):
    video, label = next(iter(data))
    return video, label

def load_batch(loader):
    videos, labels = next(iter(loader))
    return videos, labels

def predict(model, x):
    return model(x)

def main():
    batch_sizes = [2, 4, 8]
    model = VideoClassifier()
    data = TrainData(PROCESSED_DATA_PATH, model)
    sample, _  = next(iter(data))
    sample = sample[None]

    start_task("Checking load_sample()")
    repeat(load_sample, data, name="load_sample()")

    start_task("Checking predict_sample()")
    repeat(predict, model, sample, name="predict_sample()")

    for batch_size in batch_sizes:
        loader = DataLoader(data, batch_size=batch_size)
        batch, _ = next(iter(loader))

        start_task("Checking load_batch()")
        mean_load, _ =repeat(load_batch, loader, name="load_batch()")

        start_task("Checking predict_batch()")
        mean_predict, _ = repeat(predict, model, batch, name="predict_batch()")

        print(colored(f"\n> Load/ Predict (s) / Sample [{batch_size}]: {round(mean_load / batch_size, 2)}s / {round(mean_predict / batch_size, 2)}s\n", "blue"))

if __name__ == "__main__":
    main()
