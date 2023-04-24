# eval.py
#  by: mika senghaas

import os
import re
import warnings
from timeit import default_timer

import torch
from tqdm import tqdm
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import torcheval.metrics.functional as metrics
from pytorch_benchmark import benchmark

import wandb
from config import IMAGE_CLASSIFIERS, VIDEO_CLASSIFIERS, BASEPATH, RAW_DATA_PATH, DEVICE, BATCH_SIZE
from data import ImageDataset, VideoDataset
from model import ImageClassifier, VideoClassifier
from utils import end_task, load_infer_args, load_json, load_pickle, start_task

# ignore deprecation warnings in torcheval
warnings.filterwarnings("ignore", message="The reduce argument of torch.scatter with Tensor src is deprecated")

def main():
    """
    Main function for src/eval.py

    Loads a model as specified by its name and version number. The artifact and run that produced the model are
    downloaded from WANDB into the local `artifacts` directory and then loaded into memory. If the model hasn't been
    evaluated yet, the model is evaluated on the test set and the results are uploaded to WANDB. The model is then
    marked as evaluated and the artifact is updated.

    The model is evaluated using the following metrics:
    - Micro-Accuracy
    - Macro-Accuracy
    - Mirco-F1 
    - Macro-F1 
    - Hit Rate
    - Confusion Matrix
    """
    # start src/infer.py
    start = start_task("Running src/eval.py", get_timer=True)

    # load args
    args = load_infer_args()

    # load artifact from wandb
    start_task(f"Loading {args.model}:{args.version}")
    api = wandb.Api()
    artifact_path = f"mikasenghaas/bsc/{args.model}:{args.version}"
    artifact = api.artifact(artifact_path, type="model")

    # download artifact locally
    filepath = os.path.join(BASEPATH, "artifacts", f"{args.model}:{args.version}")
    if not os.path.exists(filepath):
        start_task(f"Downloaded model to {filepath}")
        artifact.download(root=filepath)

    # paths to model, transforms and config file
    model_path = os.path.join(filepath, f"{args.model}.pt")
    config_path = os.path.join(filepath, "config.json")
    transforms_path = os.path.join(filepath, "transforms.pkl")

    # load model, transforms and config
    transform = load_pickle(transforms_path)
    config = load_json(config_path)
    if args.model in IMAGE_CLASSIFIERS:
        batch_size = 32
        model = ImageClassifier(**config)
        test_data = ImageDataset(**ImageDataset.default_config())
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    elif args.model in VIDEO_CLASSIFIERS:
        batch_size = 8
        model = VideoClassifier(**config)
        test_data = VideoDataset(**VideoDataset.default_config())
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    else:
        raise Exception(f"Model {args.model} is not implemented")

    # load model state
    model.load_state_dict(torch.load(model_path))

    # get id2class and run id
    id2class = {int(i): c for i, c in config["id2class"].items()}
    num_classes = len(id2class)
    run_id = config["run_id"]

    # initialise wandb run
    start_task(f"Recognised WANB Run ID: {run_id}")
    run = api.run(f"mikasenghaas/bsc/{run_id}")

    # set eval mode
    model.eval()
    model.to("cpu")
    y_true, y_pred, y_probs = [], [], []
    samples_seen, running_time = 0, 0.0

    # benchmark model inference (FLOPs, latency, throughput, memory and energy consumption) on CPU
    start_task("Benchmarking model")
    sample, _ = next(iter(test_loader))
    bm = benchmark(model, transform(sample.to("cpu")), num_runs=10)

    benchmark_metrics = {
        "num_params": bm["params"],
        **{re.sub("batches", "samples", k): v for k, v in bm["timing"]["batch_size_1"]["on_device_inference"]["metrics"].items()},
        **bm["timing"][f"batch_size_{batch_size}"]["on_device_inference"]["metrics"],
    }

    start_task("Predicting on test split")
    for inputs, labels in tqdm(test_loader):
        with torch.no_grad():
            # move data to device
            inputs.to(DEVICE)
            labels.to(DEVICE)

            # forward pass
            logits = model(transform(inputs))

            # reshape logits
            if logits.ndim == 3:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                labels = labels.view(B*T)
                samples_seen += B*T
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
    macro_f1 = metrics.multiclass_f1_score(y_pred, y_true, average="macro", num_classes=num_classes)

    # conf_matrix
    conf_matrix = metrics.multiclass_confusion_matrix(y_pred, y_true, num_classes=num_classes)

    # inference time
    inference_time_per_sample_ms = round(running_time / samples_seen * 1000, 2)

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
    run.summary.update({"evaluated": True})

    # end src/infer.py
    end_task("src/eval.py", start)

if __name__ == "__main__":
    main()
