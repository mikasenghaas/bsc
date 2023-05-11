# eval.py
#  by: mika senghaas

import os
import warnings

import torch
from tqdm import tqdm
import pandas as pd
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import torcheval.metrics.functional as metrics
from pytorch_benchmark import benchmark

import wandb
from config import BASEPATH, WANDB_PROJECT
from utils import start_task, end_task, load_eval_args, load_json

from modules import MODULES

# ignore deprecation warnings in torcheval
warnings.filterwarnings(
    "ignore",
    message="The reduce argument of torch.scatter with Tensor src is deprecated",
)


def main():
    """
    Main function for src/eval.py

    Loads a model as specified by its name and version number. The artifact and
    run that produced the model are downloaded from WANDB into the local
    `artifacts` directory and then loaded into memory. The model is evaluated on
    the test split of the dataset and the results are uploaded to WANDB.

    The model is evaluated using the following metrics:
    - Top1-Accuracy
    - Top3-Accuracy
    - Macro-F1
    - Confusion Matrix
    """
    # start src/infer.py
    start = start_task("Running src/eval.py", get_timer=True)

    # load args
    args = load_eval_args()

    # load artifact from wandb
    start_task(f"Loading {args.model}:{args.version}")
    api = wandb.Api()
    artifact_path = f"mikasenghaas/{WANDB_PROJECT}/{args.model}:{args.version}"
    artifact = api.artifact(artifact_path, type="model")

    # download artifact locally
    filepath = os.path.join(BASEPATH, "artifacts", f"{args.model}:{args.version}")
    if not os.path.exists(filepath):
        start_task(f"Downloaded model to {filepath}")
        artifact.download(root=filepath)

    # paths to model, transforms and config file
    config_path = os.path.join(filepath, "config.json")
    model_path = os.path.join(filepath, f"{args.model}.pt")

    # load run config
    config = load_json(config_path)
    model_type = config["general"]["type"]
    run_id = config["wandb"]["run_id"]
    batch_size = config["loader"]["batch_size"]
    device = config["trainer"]["device"] if not args.device else args.device

    # initialise transform and model
    TRANSFORM = MODULES[model_type]["transform"]
    MODEL = MODULES[model_type]["model"]
    DATA = MODULES[model_type]["data"]

    # load transform, model, data
    transform = TRANSFORM(**config["transform"])
    model = MODEL(**config["model"])
    test_data = DATA(**config["dataset"], split="test", transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # load test data (all frames)
    class2id = test_data.class2id
    num_classes = len(class2id)

    # load model state
    start_task("Loading Weights")
    model.load_state_dict(torch.load(model_path))

    # initialise wandb run
    start_task(f"Recognised WANB Run ID: {run_id}")
    run = api.run(f"mikasenghaas/{WANDB_PROJECT}/{run_id}")

    # set eval mode
    model.eval()
    model.to("cpu")
    y_true, y_pred, y_probs = [], [], []
    samples_seen = 0

    # benchmark model inference on CPU
    start_task("Benchmarking model")
    sample = next(iter(test_loader))
    benchmark_metrics = {}
    match model_type:
        case "image":
            bm = benchmark(model, sample[0].to("cpu"), num_runs=5)
        case "video":
            if args.model != "slowfast_r50":
                bm = benchmark(model, sample["video"].to("cpu"), num_runs=5)
                benchmark_metrics = pd.json_normalize(bm, sep="_").to_dict(
                    orient="records"
                )[0]

    start_task("Predicting on test split")
    model.to(device)
    for batch in tqdm(test_loader):
        with torch.no_grad():
            match model_type:
                case "image":
                    inputs, labels = batch
                case "video":
                    inputs = batch["video"]
                    labels = batch["label"]
                    labels = torch.tensor(
                        [class2id[label] for label in labels], dtype=torch.long
                    )
                case _:
                    raise ValueError(f"Model type {model_type} not supported.")

            # move inputs to device
            match args.model:
                case "slowfast_r50":
                    inputs = [i.to(device) for i in inputs]
                case _:
                    inputs = inputs.to(device)

            # move labels to device
            labels = labels.to(device)

            # forward pass
            logits = model(inputs)

            # reshape logits
            if logits.ndim == 3:
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                labels = labels.view(B * T)
                samples_seen += B * T
            else:
                B, C = logits.shape
                samples_seen += B

            # compute probs and preds
            probs = softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # append to lists
            y_probs = y_probs + probs.tolist()
            y_pred = y_pred + preds.tolist()
            y_true = y_true + labels.tolist()

    start_task("Computing performance metrics")
    # convert to torch tensors
    y_probs = torch.tensor(y_probs)
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)

    # compute multiclass accuracy
    top1_acc = metrics.multiclass_accuracy(y_pred, y_true, k=1)
    top3_acc = metrics.multiclass_accuracy(y_probs, y_true, k=3)

    # compute f1 score
    macro_f1 = metrics.multiclass_f1_score(
        y_pred, y_true, average="macro", num_classes=num_classes
    )

    # conf_matrix
    conf_matrix = metrics.multiclass_confusion_matrix(
        y_pred, y_true, num_classes=num_classes
    )

    # create dict with metrics
    eval_metrics = {
        "test_top1_acc": top1_acc.item(),
        "test_top3_acc": top3_acc.item(),
        "test_macro_f1": macro_f1.item(),
        "test_conf_matrix": conf_matrix.tolist(),
    }

    # merge dicts with metrics
    all_metrics = {**benchmark_metrics, **eval_metrics}

    # log metrics to wandb
    start_task("Syncing to WANDB")
    run.summary.update(all_metrics)

    # end src/infer.py
    end_task("src/eval.py", start)


if __name__ == "__main__":
    main()
