# train.py
#  by: mika senghaas

import argparse
import os
from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile
from tqdm import tqdm

import wandb
from config import MODEL_PATH, SPLITS
from data import ImageDataset
from model import FinetunedImageClassifier
from transform import ImageTransformer
from utils import (
    end_task,
    get_progress_bar,
    get_summary,
    load_train_args,
    mkdir,
    save_json,
    save_pickle,
    start_task,
    unnormalise_image,
)


def train(
    model: nn.Module,
    transform: ImageTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optim: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
):
    """
    Train a model on a given dataset given a data transformer class, loader classes for
    a training and validation split, an criterion, optimizer and learning rate
    scheduler.

    Args:
        model (nn.Module): Model to be trained
        transform (ImageTransformer): Data transformer class
        train_loader (DataLoader): Training split loader
        val_loader (DataLoader): Validation split loader
        criterion (nn.Module): Loss function
        optim (torch.optim): Optimizer
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        args (argparse.Namespace): Arguments
    """
    # set progress bar with max epochs
    pbar = tqdm(range(args.max_epochs))
    pbar.set_description(
        "XX/XX | XX/XX | XX.Xms/ XX.Xms | Train: X.XXX (XX.X%) | Val: X.XXX (XX.X%)"
    )

    # initialise training metrics
    train_loss, val_loss = 0.0, 0.0
    train_acc, val_acc = 0.0, 0.0

    # initialise arrays to keep track of training and inference speed
    training_times, inference_times = [], []

    # put model on device
    model.to(args.device)

    # start trainingj
    for epoch in pbar:
        # initialise running metrics
        running_loss, running_correct = 0.0, 0
        running_training_time, running_inference_time = 0.0, 0.0

        # set model to train mode
        model.train()

        # iterate over training data
        for batch_num, (inputs, labels) in enumerate(train_loader):
            # put data on device
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # zero the parameter gradients
            optim.zero_grad()

            # forward pass
            start = default_timer()
            logits = model(transform(inputs))
            running_inference_time += default_timer() - start

            # compute predictions
            preds = torch.argmax(logits, 1)

            # compute loss
            loss = criterion(logits, labels)

            # backprop error
            loss.backward()
            optim.step()

            running_training_time += default_timer() - start

            # performance metrics
            running_loss += loss.item()
            running_correct += torch.sum(preds == labels).item()
            samples_seen = (batch_num + 1) * args.batch_size

            # normalise
            train_acc = running_correct / samples_seen
            train_loss = running_loss / samples_seen

            progress = get_progress_bar(
                epoch,
                args.max_epochs,
                batch_num,
                len(train_loader),
                running_training_time,
                running_inference_time,
                samples_seen,
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
                        "training_accuracy": train_acc,
                        "validation_accuracy": val_acc,
                        "training_loss": train_loss,
                        "validation_loss": val_loss,
                    }
                )

        training_times.append(running_training_time)
        inference_times.append(running_inference_time)

        if val_loader is not None:
            # initialise running metrics (not tracking time)
            running_loss, running_correct = 0.0, 0

            # set model to eval mode
            model.eval()

            # iterate over validation data
            for batch_num, (inputs, labels) in enumerate(val_loader):
                # put data on device
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                # forward pass
                logits = model(transform(inputs))
                preds = torch.argmax(logits, 1)
                loss = criterion(logits, labels)

                # accumulate loss and correct predictions
                running_loss += loss.item()
                running_correct += torch.sum(labels == preds).item()

            val_loss = running_loss / len(val_loader.dataset)  # type: ignore
            val_acc = running_correct / len(val_loader.dataset)  # type: ignore

            progress = get_progress_bar(
                epoch,
                args.max_epochs,
                batch_num,  # type: ignore
                len(val_loader),
                running_training_time,
                running_inference_time,
                len(val_loader.dataset),  # type: ignore
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )
            pbar.set_description(progress)

        # adjust learning rate
        scheduler.step()

    # log average training step time/ sample + inference time/ sample
    training_time_per_sample = round(sum(training_times) / len(training_times), 1)
    inference_time_per_sample = round(sum(inference_times) / len(inference_times), 1)
    if args.wandb_log:
        wandb.config.update(
            {
                "training_time_per_sample_ms": training_time_per_sample,
                "inference_time_per_sample_ms": inference_time_per_sample,
            }
        )

    return model


def main():
    """Main function of src/train.py"""
    start_timer = start_task("Running train.py", get_timer=True)

    # parse cli arguments
    args = load_train_args()

    # initialise wandb run
    if args.wandb_log:
        wandb.init(
            project="bsc",
            group=args.wandb_group if args.wandb_group else None,
            name=args.wandb_name if args.wandb_name else None,
            tags=args.wandb_tags if args.wandb_tags else None,
            config=vars(args),
        )

        wandb.define_metric("training_loss", summary="min")
        wandb.define_metric("validation_loss", summary="min")
        wandb.define_metric("training_accuracy", summary="max")
        wandb.define_metric("validation_accuracy", summary="max")

    # load data
    start_task("Initialising Data and Model")

    # initialise train and validation split
    data = {
        split: ImageDataset(
            split=split, include_classes=args.include_classes, ratio=args.ratio
        )
        for split in ["train", "val"]
    }

    # extract class2id and id2class mappings from train split
    id2class, class2id = data["train"].id2class, data["train"].class2id

    # initialise test split with mappings
    data["test"] = ImageDataset(
        split="test",
        include_classes=args.include_classes,
        ratio=args.ratio,
        class2id=class2id,
        id2class=id2class,
    )

    # initialise transforms
    transform = ImageTransformer()

    # initialise model
    model = FinetunedImageClassifier(
        model_name=args.model,
        num_classes=len(args.include_classes),
        pretrained=args.pretrained,
        id2class=id2class,
        class2id=class2id,
    )

    # initialise data loader
    loader = {
        split: DataLoader(data[split], batch_size=args.batch_size) for split in SPLITS
    }

    # define loss, optimiser and lr scheduler
    criterion = nn.CrossEntropyLoss()  # pyright: ignore
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, args.step_size, args.gamma)

    # train model
    start_task("Starting Training")
    print(get_summary(vars(args)))
    trained_model = train(
        model,
        transform,
        loader["train"],
        loader["val"],
        criterion,
        optim,
        scheduler,
        args,
    )

    # prepare trained model for saving
    trained_model.to("cpu")
    trained_model.eval()

    if args.wandb_log:
        # prepare artifact saving
        filepath = os.path.join(MODEL_PATH, args.model)
        mkdir(filepath)

        # optimised model
        start_task("Optimising Model for Mobile")
        torchscript_model = torch.jit.script(trained_model)  # type: ignore
        optimised_torchscript_model = optimize_for_mobile(
            torchscript_model  # type: ignore
        )

        # save transforms and model
        start_task(f"Saving Artifacts to {filepath}")
        save_pickle(transform, os.path.join(filepath, "transforms.pkl"))
        save_json(trained_model.meta, os.path.join(filepath, "config.json"))
        torch.save(
            trained_model.state_dict(),
            os.path.join(filepath, f"{args.model}.pt"),
        )
        optimised_torchscript_model.save(
            os.path.join(filepath, f"{args.model}.pth")
        )  # type: ignore
        optimised_torchscript_model._save_for_lite_interpreter(
            os.path.join(filepath, f"{args.model}.ptl")
        )  # type: ignore

        # save as artifact to wandb
        start_task("Saving Artifcats to WANDB")
        artifact = wandb.Artifact(args.model, type="model")
        artifact.add_dir(filepath)
        wandb.log_artifact(artifact)

        start_task("Evaluating Model")
        # iterate over test set and put predictions and true labels in two numpy arrays
        y_true, y_pred = [], []
        for batch_num, (inputs, labels) in enumerate(loader["test"]):
            # predict test samples
            logits = trained_model(transform(inputs))
            preds = logits.argmax(-1)

            # append to lists
            y_true += list(labels.cpu())
            y_pred += list(preds.cpu())

        # convert to numpy arrays
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        # compute test accuracy
        test_accuracy = np.mean(y_true == y_pred)

        # save test accuracy to wandb
        wandb.run.summary["test_accuracy"] = test_accuracy  # type: ignore

        # compute mispredicted images
        mispredictions = []
        for batch_num, (images, _) in enumerate(loader["test"]):
            for i, image in enumerate(images):
                idx = batch_num * i + i
                true, pred = y_true[idx], y_pred[idx]  # pyright: ignore
                if true != pred and len(mispredictions) < 20:
                    image = unnormalise_image(image)
                    caption = f"True: {id2class[true]} / Pred: {id2class[pred]}"
                    wandb_img = wandb.Image(image, caption=caption)

                    mispredictions.append(wandb_img)

        # save evaluation visualisations to wandb
        wandb.log({"mispredictions": mispredictions})

    end_task("Training Done", start_timer)


if __name__ == "__main__":
    main()
