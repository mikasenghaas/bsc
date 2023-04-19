# train.py
#  by: mika senghaas

import argparse
import os
from timeit import default_timer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile
from tqdm import tqdm
import wandb

from config import IMAGE_CLASSIFIERS, VIDEO_CLASSIFIERS, ARTIFACTS_PATH
from data import ImageDataset, VideoDataset
from model import ImageClassifier, VideoClassifier
from transform import FrameTransformer
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
    transform: FrameTransformer,
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
        transform (FrameTransformer): Data transformer class
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
    best_train_acc, best_val_acc = 0.0, 0.0
    training_times = []

    # put model on device
    model.to(args.device)

    # start trainingj
    for epoch in pbar:
        # initialise running metrics
        running_loss, running_correct = 0.0, 0
        running_time = 0.0
        samples_seen = 0

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
            running_time += default_timer() - start

            # if sequence prediction, reshape logits
            if logits.ndim == 3:
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                labels = labels.view(B * T)
                samples_seen += B * T
            elif logits.ndim == 2:
                B, C = logits.shape
                samples_seen += B

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
            train_acc = running_correct / samples_seen
            train_loss = running_loss / samples_seen
            training_time = running_time / samples_seen

            if train_acc > best_train_acc:
                best_train_acc = train_acc

            progress = get_progress_bar(
                epoch + 1,
                args.max_epochs,
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
                        "training_accuracy": train_acc,
                        "training_loss": train_loss,
                    }
                )

        training_times.append(training_time)

        if val_loader is not None:
            # initialise running metrics (not tracking time)
            running_loss, running_correct = 0.0, 0
            samples_seen = 0

            # set model to eval mode
            model.eval()

            # iterate over validation data
            for batch_num, (inputs, labels) in enumerate(val_loader):
                # put data on device
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                # forward pass
                logits = model(transform(inputs))

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
                args.max_epochs,
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
                        "validation_accuracy": val_acc,
                        "validation_loss": val_loss,
                    }
                )

        # adjust learning rate
        scheduler.step()

    # log average training step time/ sample
    training_time_per_sample_ms = round(
        sum(training_times) * 1000 / len(training_times), 2
    )
    if args.wandb_log:
        wandb.summary.update(
            {
                "training_time_per_sample_ms": training_time_per_sample_ms,
                "best_training_accuracy": best_train_acc,
                "best_validation_accuracy": best_val_acc,
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
        run = wandb.init(
            project="bsc",
            group=args.wandb_group if args.wandb_group else None,
            name=args.wandb_name if args.wandb_name else None,
            tags=args.wandb_tags if args.wandb_tags else None,
            config=vars(args),
        )

    # load data
    start_task("Initialising Data and Model")

    # image or video classifier
    if args.model in IMAGE_CLASSIFIERS:
        start_task(f"Recognised ImageClasssifier {args.model}")
        train_data = ImageDataset(
            split="train", include_classes=args.include_classes, ratio=args.ratio
        )
        test_data = ImageDataset(
            split="test", include_classes=args.include_classes, ratio=args.ratio
        )

        model = ImageClassifier(
            model_name=args.model,
            num_classes=len(args.include_classes),
            id2class=train_data.id2class,
            class2id=train_data.id2class,
            run_id=run.id if args.wandb_log else None,
        )

    elif args.model in VIDEO_CLASSIFIERS:
        start_task(f"Recognised VideoClassifier {args.model}")
        train_data = VideoDataset(
            split="train", include_classes=args.include_classes, ratio=args.ratio
        )
        test_data = VideoDataset(
            split="test", include_classes=args.include_classes, ratio=args.ratio
        )

        model = VideoClassifier(
            model_name=args.model,
            num_classes=len(args.include_classes),
            id2class=train_data.id2class,
            class2id=train_data.id2class,
            run_id=run.id if args.wandb_log else None,
        )
    else:
        raise ValueError(f"Unrecognised model {args.model}")

    # initialise data loader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # initialise transforms
    transform = FrameTransformer()

    # define loss, optimiser and lr scheduler
    criterion = nn.CrossEntropyLoss()  # pyright: ignore
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, args.step_size, args.gamma)

    # train model
    start_task("Starting Training")

    # print training configuration
    print("\nTraining Configuration:")
    print(get_summary(vars(args)))

    trained_model = train(
        model,
        transform,
        train_loader,
        test_loader,
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
        save_json(trained_model.meta, os.path.join(filepath, "config.json"))
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
