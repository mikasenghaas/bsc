# train.py
#  by: mika senghaas

import datetime

import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from data import TrainData
from model import VideoClassifier
from utils import *

def main():
    start_timer = start_task("Running train.py", get_timer=True)

    # load model and dataset
    start_task("Initialising Model")
    model = VideoClassifier()

    start_task("Initialising Training Data")
    data = TrainData(filepath=PROCESSED_DATA_PATH, model=model)
    # prior: one clip: 2.408.448 float32 = 77.070.336 Bits = 9 MB

    loader = DataLoader(data, batch_size=BATCH_SIZE)

    # load conversions
    label2id, id2label = model.label2id, model.id2label

    # optimiser and
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # start training
    start_task("Starting Training Loop")
    for epoch in range(MAX_EPOCHS):
        pbar = tqdm(loader)
        batch = 0
        pbar.set_description(f"Epoch {epoch+1}/{MAX_EPOCHS} - Batch {batch}/{len(loader)} - 0.000")
        for batch, (x, y) in enumerate(pbar):
            # x: (B, T, C, H, W); y: B

            # predict class
            logits = model(x)
            probs = softmax(logits)

            # compute loss
            loss = criterion(probs, y)
            loss.backward()
            optim.step()

            pbar.set_description(f"Epoch {epoch+1}/{MAX_EPOCHS} - Batch {batch+1}/{len(loader)} - {round(loss.item(), 3)}")
            batch += 1


    start_task("Saving Model")
    mkdir(MODEL_PATH)
    save_path = f"{MODEL_PATH}/{datetime.datetime.now()}"
    torch.save(model.state_dict(), save_path)

    end_task("Training Done", start_timer)

if __name__ == '__main__':
  main()
