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

from transform import ImageTransform, VideoTransform
from defaults import DEFAULT
from modules import MODULES

from config import WANDB_PROJECT, ARTIFACTS_PATH

from utils import (
    end_task,
    get_progress_bar,
    get_summary,
    load_train_args,
    mkdir,
    save_json,
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
    model_name = config["general"]["name"]
    model_type = config["general"]["type"]
    batch_size = config["loader"]["batch_size"]
    epochs = config["trainer"]["epochs"]
    device = config["trainer"]["device"]
    class2id = train_loader.dataset.class2id

    pbar = tqdm(range(epochs))
    pbar.set_description(
        "XX/XX | XX/XX | XX.Xms/ XX.Xms | Train: X.XXX (XX.X%) | Val: X.XXX (XX.X%)"
    )

    # initialise training metrics
    train_loss, val_loss = 0.0, 0.0
    train_acc, val_acc = 0.0, 0.0
    best_train_acc, best_val_acc = 0.0, 0.0
    frames_per_epoch = 0  # unknown in advance because of streaming data

    # put model on device
    model.to(device)

    # start trainingj
    for epoch in pbar:
        # initialise running metrics
        running_loss, running_correct = 0.0, 0
        running_time = 0.0
        frames_seen = 0  # sample is one prediction
        samples_seen = 0  # sample is one prediction

        # set model to train mode
        model.train()

        # iterate over training data
        for batch_num, batch in enumerate(train_loader):
            # get inputs and labels
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

            # put inputs on device
            match model_name:
                case "SlowFast R50":
                    inputs = [i.to(device) for i in inputs]
                case _:
                    inputs = inputs.to(device)

            # put labels on device
            labels = labels.to(device)

            # zero the parameter gradients
            optim.zero_grad(set_to_none=True)

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

            # compute frames seen
            match model_type:
                case "image":
                    frames_seen = samples_seen
                case "video":
                    frames_seen = samples_seen * config["transform"]["num_frames"]

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
            train_acc = running_correct / samples_seen  # acc per prediction
            train_loss = running_loss / samples_seen  # loss per prediction
            training_time = running_time / frames_seen  # time per frame

            if train_acc > best_train_acc:
                best_train_acc = train_acc

            progress = get_progress_bar(
                epoch + 1,
                epochs,
                batch_num + 1,
                training_time,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                train=True,
            )
            pbar.set_description(progress)

            # log epoch metrics for train and val split
            if args.wandb_log:
                wandb.log(
                    {
                        "frames_seen": frames_per_epoch * epoch + frames_seen,
                        "train/acc": train_acc,
                        "train/loss": train_loss,
                    }
                )

        frames_per_epoch = frames_seen

        # list of k random integers between 0 and len(train_data)
        match model_type:
            case "image":
                k = int(len(test_data) * 0.05)
                random_indices = torch.randint(0, len(test_data), (k,))
                val_data = Subset(test_data, random_indices)
                val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
            case "video":
                val_loader = DataLoader(test_data, batch_size=batch_size)
            case _:
                raise ValueError(f"Model type {model_type} not supported.")

        # initialise running metrics (not tracking time)
        running_loss, running_correct = 0.0, 0
        samples_seen = 0

        # set model to eval mode
        model.eval()

        # iterate over validation data
        for batch_num, batch in enumerate(val_loader):
            # get inputs and labels
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

            # put inputs on device
            match model_name:
                case "SlowFast R50":
                    inputs = [i.to(device) for i in inputs]
                case _:
                    inputs = inputs.to(device)

            # put labels on device
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

            val_acc = running_correct / samples_seen
            val_loss = running_loss / samples_seen

            progress = get_progress_bar(
                epoch + 1,
                epochs,
                batch_num + 1,
                training_time,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                train=False,
            )
            pbar.set_description(progress)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

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
    config["trainer"]["device"] = (
        args.device if args.device else config["trainer"]["device"]
    )
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
    criterion = nn.CrossEntropyLoss(reduction="sum")  # pyright: ignore
    optim = torch.optim.AdamW(model.parameters(), **config["optim"])

    # initialise wandb run
    start_task("Initialising W&B")
    if args.wandb_log:
        wandb.init(
            project=WANDB_PROJECT,
            group=args.wandb_group if args.wandb_group else None,
            name=args.wandb_name if args.wandb_name else None,
            tags=args.wandb_tags if args.wandb_tags else None,
            config=config,
        )

        # set custom x axis for logging
        wandb.define_metric("samples_seen")
        wandb.define_metric("frames_seen")
        # set all train metrics to use this step
        wandb.define_metric("train/*", step_metric="frames_seen")
        wandb.define_metric("val/*", step_metric="frames_seen")

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

        # add wandb metadata to config
        config["wandb"] = {
            "run_id": wandb.run.id,
            "run_url": wandb.run.get_url(),
            "run_name": wandb.run.name,
            "run_group": wandb.run.group,
            "run_tags": wandb.run.tags,
        }

        # save transforms and model
        start_task(f"Saving Artifacts to {filepath}")
        save_json(config, os.path.join(filepath, "config.json"))
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
