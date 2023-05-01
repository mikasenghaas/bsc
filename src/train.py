# train.py
#  by: mika senghaas
import argparse
import os
from timeit import default_timer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import wandb

from data import ImageDataset, VideoDataset
from transform import ImageTransform, VideoTransform
from defaults import DEFAULT
from modules import MODULES

from config import ARTIFACTS_PATH

from utils import (
    end_task,
    get_progress_bar,
    get_summary,
    load_train_args,
    mkdir,
    save_json,
    save_pickle,
    start_task,
)


def train(
    model: nn.Module,
    transform: ImageTransform or VideoTransform,
    train_loader: DataLoader,
    test_data: Dataset,
    criterion: nn.Module,
    optim: torch.optim.Optimizer,
    args: argparse.Namespace,
    config,
):
    """
    Train a model on a given dataset given a data transformer class, loader classes for
    a training and validation split, an criterion, optimizer.

    Args:
        model (nn.Module): Model to be trained
        transform (FrameTransformer): Data transformer class
        train_loader (DataLoader): Training split loader
        val_loader (DataLoader): Validation split loader
        criterion (nn.Module): Loss function
        optim (torch.optim): Optimizer
        args (argparse.Namespace): Arguments
    """
    # set progress bar with max epochs
    batch_size = config["loader"]["batch_size"]
    epochs = config["trainer"]["epochs"]
    device = config["trainer"]["device"]

    pbar = tqdm(range(epochs))
    pbar.set_description(
        "XX/XX | XX/XX | XX.Xms/ XX.Xms | Train: X.XXX (XX.X%) | Val: X.XXX (XX.X%)"
    )

    # initialise training metrics
    train_loss, val_loss = 0.0, 0.0
    train_acc, val_acc = 0.0, 0.0
    best_train_acc, best_val_acc = 0.0, 0.0
    training_times = []

    # compute total number of samples
    train_data = train_loader.dataset
    if type(train_data) == ImageDataset:
        total_samples = len(train_data)
    elif type(train_data) == VideoDataset:
        num_frames = config["transform"]["num_frames"]
        total_samples = (
            len(train_data) * num_frames * batch_size
        )  # batches * clips/batch * frames/clip
    else:
        raise ValueError("Unknown dataset type")
    # print(total_samples)

    # put model on device
    model.to(device)

    # start trainingj
    for epoch in pbar:
        # initialise running metrics
        running_loss, running_correct = 0.0, 0
        running_time = 0.0
        samples_seen = 0 # sample is one prediction
        frames_seen = 0 # frame is one sample

        # set model to train mode
        model.train()

        # iterate over training data
        for batch_num, (inputs, labels) in enumerate(train_loader):
            # put data on device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(inputs.shape)
            # print(labels.shape)

            # zero the parameter gradients
            optim.zero_grad()

            # forward pass
            start = default_timer()
            output = model(inputs)
            running_time += default_timer() - start

            # google lenet returns custom output object
            if type(output) != torch.Tensor:
                logits = output.logits
            else:
                logits = output

            # reshape logits and update samples seen
            if logits.ndim == 3:
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                labels = labels.view(B * T)
                samples_seen += B * T
            elif logits.ndim == 2:
                B, C = logits.shape
                samples_seen += B

            # compuute frames seen
            if inputs.ndim == 4: # image
                frames_seen = samples_seen
            elif inputs.ndim == 5: # video
                B, _, T, _, _ = inputs.shape
                frames_seen += B * T

            # compute predictions
            preds = torch.argmax(logits, 1)

            # compute loss
            loss = criterion(logits, labels)

            # backprop error
            loss.backward()
            optim.step()

            # performance metrics
            running_loss += loss.item()
            running_correct += torch.sum(preds == labels).item()

            # normalise
            train_acc = running_correct / samples_seen # loss per prediction
            train_loss = running_loss / samples_seen # acc per prediction
            training_time = running_time / frames_seen # time per frame

            if train_acc > best_train_acc:
                best_train_acc = train_acc

            progress = get_progress_bar(
                epoch + 1,
                epochs,
                batch_num + 1,
                len(train_loader),
                training_time,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )
            pbar.set_description(progress)

            # log epoch metrics for train and val split
            if args.wandb_log:
                wandb.log(
                    {
                        "frames_seen": epoch * total_samples + frames_seen,
                        "train/acc": train_acc,
                        "train/loss": train_loss,
                    }
                )

        training_times.append(training_time)

        # list of k random integers between 0 and len(train_data)
        k = int(len(test_data) * 0.05)
        random_indices = torch.randint(0, len(test_data), (k,))
        val_data = Subset(test_data, random_indices)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        # initialise running metrics (not tracking time)
        running_loss, running_correct = 0.0, 0
        samples_seen = 0

        # set model to eval mode
        model.eval()

        # iterate over validation data
        for batch_num, (inputs, labels) in enumerate(val_loader):
            # put data on device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward pass
            logits = model(inputs)

            # if sequence prediction, reshape logits
            if logits.ndim == 3:
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                labels = labels.view(B * T)
                samples_seen += B * T

            elif logits.ndim == 2:
                B, C = logits.shape
                samples_seen += B

            # compute predictions and loss
            preds = torch.argmax(logits, 1)
            loss = criterion(logits, labels)

            # accumulate loss and correct predictions
            running_loss += loss.item()
            running_correct += torch.sum(labels == preds).item()

        val_loss = running_loss / samples_seen
        val_acc = running_correct / samples_seen

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        progress = get_progress_bar(
            epoch + 1,
            epochs,
            batch_num + 1,
            len(val_loader),
            training_time,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )
        pbar.set_description(progress)

        # log epoch metrics for train and val split
        if args.wandb_log:
            wandb.log(
                {
                    "val/acc": val_acc,
                    "val/loss": val_loss,
                }
            )

    if args.wandb_log:
        wandb.summary.update(
            {
                "train/best_acc": best_train_acc,
                "val/best_acc": best_val_acc,
            }
        )

    return model


def main():
    """Main function of src/train.py"""
    start_timer = start_task("Running train.py", get_timer=True)

    # parse cli arguments
    args = load_train_args()

    # load config
    start_task(f"Loading configuation for {args.model}")
    config = DEFAULT[args.model]

    # overwrite config with cli arguments (if any)
    config["trainer"]["epochs"] = (
        args.epochs if args.epochs else config["trainer"]["epochs"]
    )
    config["optim"]["lr"] = args.lr if args.lr else config["optim"]["lr"]
    config["loader"]["batch_size"] = (
        args.batch_size if args.batch_size else config["loader"]["batch_size"]
    )

    # extract model type
    model_type = config["general"]["type"]
    pp = model_type[0].upper() + model_type[1:]

    # initialise data
    start_task(f"Initialising {pp}Dataset, {pp}Transform and {pp}Classifier")
    DATA = MODULES[model_type]["data"]
    TRANSFORM = MODULES[model_type]["transform"]
    MODEL = MODULES[model_type]["model"]

    # load transform, model, data
    transform = TRANSFORM(**config["transform"])
    train_data = DATA(**config["dataset"], split="train", transform=transform)
    test_data = DATA(**config["dataset"], split="test", transform=transform)
    model = MODEL(**config["model"])

    # initialise data loader
    start_task("Initialising data loader")
    train_loader = DataLoader(train_data, **config["loader"])

    # define loss, optimiser
    criterion = nn.CrossEntropyLoss()  # pyright: ignore
    optim = torch.optim.AdamW(model.parameters(), **config["optim"])

    # initialise wandb run
    start_task("Initialising W&B")
    if args.wandb_log:
        wandb.init(
            project="bsc-2",
            group=args.wandb_group if args.wandb_group else None,
            name=args.wandb_name if args.wandb_name else None,
            tags=args.wandb_tags if args.wandb_tags else None,
            config=config,
        )

        # set custom x axis for logging
        wandb.define_metric("samples_seen")
        # set all train metrics to use this step
        wandb.define_metric("train/*", step_metric="samples_seen")
        wandb.define_metric("val/*", step_metric="samples_seen")

    # train model
    start_task("Starting Training")

    # print training configuration
    print("\nTraining Configuration:")
    print(get_summary(config))
    print()

    trained_model = train(
        model,
        transform,
        train_loader,
        test_data,
        criterion,
        optim,
        args,
        config,
    )

    # set to eval and cpu mode for saving
    trained_model.to("cpu")
    trained_model.eval()

    if args.wandb_log:
        # prepare artifact saving
        filepath = os.path.join(ARTIFACTS_PATH, f"{args.model}:latest")
        mkdir(filepath)

        # optimised model
        """
        start_task("Optimising Model for Mobile")
        torchscript_model = torch.jit.script(trained_model)  # type: ignore
        optimised_torchscript_model = optimize_for_mobile(
            torchscript_model  # type: ignore
        )
        """

        # save transforms and model
        start_task(f"Saving Artifacts to {filepath}")
        save_pickle(transform, os.path.join(filepath, "transforms.pkl"))
        torch.save(
            trained_model.state_dict(),
            os.path.join(filepath, f"{args.model}.pt"),
        )
        """
        optimised_torchscript_model.save(
            os.path.join(filepath, f"{args.model}.pth")
        )  # type: ignore
        optimised_torchscript_model._save_for_lite_interpreter(
            os.path.join(filepath, f"{args.model}.ptl")
        )  # type: ignore
        """

        # save as artifact to wandb
        start_task("Saving Artifcats to WANDB")
        artifact = wandb.Artifact(args.model, type="model")
        artifact.add_dir(filepath)
        wandb.log_artifact(artifact)

    end_task("Training Done", start_timer)


if __name__ == "__main__":
    main()
