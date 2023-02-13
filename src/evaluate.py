# evaluate.py
#  by: mika senghaas

import numpy as np
from torch.utils.data import DataLoader

from config import *
from data import TrainData
from model import VideoClassifier
from utils import *

def main():
    # get all saved models
    models = os.listdir(MODEL_PATH)
    sorted_models = sorted(models)
    most_recent = sorted_models[-1]

    # load model from most recent checkpoint
    model = VideoClassifier()
    model.load_state_dict(torch.load(f"{MODEL_PATH}/{most_recent}"))

    # set model into evaluation mode
    model.eval()

    # load data and mini-batch data
    data = TrainData(filepath=PROCESSED_DATA_PATH, model=model)
    loader = DataLoader(data, batch_size=4) 

    # load and predict on test split
    predicted_labels = []
    true_labels = []
    for x, y in loader:
        # predict on all samples
        logits = model(x)
        preds = logits.argmax(-1)

        for i in range(len(y)):
            predicted_labels.append(preds[i].item())
            true_labels.append(y[i].item())

    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)

    accuracy_score = np.mean(predicted_labels == true_labels)
    print(f"Accuracy Score: {round(100 * accuracy_score, 2)}%")

if __name__ == "__main__":
    main()
