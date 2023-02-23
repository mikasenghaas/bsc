# evaluate.py
#  by: mika senghaas

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

from sklearn.metrics import classification_report, confusion_matrix

from config import *
from utils import *
from data import *
from model import *

def main():
    # get parser
    parser = load_evaluate_parser()
    args = parser.parse_args()

    # get model and data
    match args.model:
        case 'resnet':
            test_data = ImageDataset(PROCESSED_DATA_PATH, split="test")
            model = ResNet(test_data.num_classes)
        case _:
            # default case
            test_data = ImageDataset(PROCESSED_DATA_PATH, split="test")
            model = ResNet(test_data.num_classes)

    # get filepath
    start_task(f"Loading {args.model}")
    if not args.filepath:
        filepaths = sorted(glob.glob(os.path.join(MODEL_PATH, args.model, "*")))
        filepath = filepaths[-1]
    else:
        filepath = args.filepath

    # load model
    try:
        model.load_state_dict(torch.load(filepath))
    except:
        raise Exception(f"No trained model found at {filepath}")
    start_task(f"Found at {filepath}")

    # set model into evaluation mode
    model.to(DEVICE)
    model.eval()

    # load data and mini-batch data
    loader = DataLoader(test_data, batch_size=BATCH_SIZE) 

    # load and predict on test split
    y_pred = np.empty(0)
    y_true = np.empty(0)
    for inputs, labels in loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # predict test samples
        logits = model(inputs)
        preds = logits.argmax(-1)

        y_pred = np.concatenate((y_pred, preds.cpu()))
        y_true = np.concatenate((y_true, labels.cpu()))

    # compute classification report and confusion matrix
    report = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(report)
    sns.heatmap(pd.DataFrame(conf_matrix), annot=True)
    plt.show()

if __name__ == "__main__":
    main()
