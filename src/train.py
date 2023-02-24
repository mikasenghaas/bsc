# train.py
#  by: mika senghaas

import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from config import *
from data import ImageDataset
from model import ResNet
from utils import *

def train(model, train_loader, val_loader, criterion, optim, scheduler, args):
    model.to(args.device)
    pbar = tqdm(range(args.max_epochs))
    pbar.set_description(f'XXX/XX - Train: X.XXX (XX.X%) - Val: X.XXX (XX.X%)')
    train_loss, val_loss = 0.0, 0.0
    train_acc, val_acc = 0.0, 0.0
    for epoch in pbar:
        running_loss, running_correct = 0.0, 0
        model.train()
        for batch_num, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
  
            # zero the parameter gradients
            optim.zero_grad()
  
            logits = model(inputs)
            preds = torch.argmax(logits, 1)
            loss = criterion(logits, labels)
  
            # backward + optimize only if in training phase
            loss.backward()
            optim.step()
            scheduler.step()

            # performance metrics
            running_loss += loss.item()
            running_correct += torch.sum(preds == labels)
            samples_seen = (batch_num + 1) * args.batch_size

            # normalise
            train_acc = running_correct / samples_seen
            train_loss = running_loss / samples_seen
            
            pbar.set_description(f'{str(epoch).zfill(len(str(args.max_epochs)))}/{str(batch_num).zfill(len(str(len(train_loader))))} - Train: {train_loss:.3f} ({(train_acc * 100):.1f}%) - Val: {val_loss:.3f} ({(val_acc * 100):.1f}%)')

            # log epoch metrics for train and val split
            if args.log:
                wandb.log({
                    'Training Accuracy': train_acc, 
                    'Validation Accuracy': val_acc,
                    'Training Loss': train_loss, 
                    'Validation Loss': val_loss,
                    'Epoch': epoch,
                    'Batch': batch_num})
                
        if val_loader != None:
            running_loss, running_correct = 0.0, 0
            model.eval()
            for batch_num, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
      
                logits = model(inputs)
                preds = torch.argmax(logits, 1)
                loss = criterion(logits, labels)

                # accumulate loss and correct predictions
                running_loss += loss.item()
                running_correct += torch.sum(labels == preds)

            val_loss = running_loss / len(val_loader.dataset)
            val_acc = running_correct / len(val_loader.dataset)

            pbar.set_description(f'{str(epoch).zfill(len(str(args.max_epochs)))}/00 - Train: {train_loss:.3f} ({(train_acc * 100):.1f}%) - Val: {val_loss:.3f} ({(val_acc * 100):.1f}%)')

    return model

def main():
    start_timer = start_task("Running train.py", get_timer=True)

    # parse cli arguments
    args = load_train_args()

    # initialise wandb run
    if args.log:
        wandb.init(project="bsc", notes="Computer Vision Models for Indoor Localisation", config=vars(args))

        wandb.define_metric("Training Loss", summary="min")
        wandb.define_metric("Validation Loss", summary="min")
        wandb.define_metric("Training Accuracy", summary="max")
        wandb.define_metric("Validation Accuracy", summary="max")

    # load data 
    start_task("Initialising Data and Model")
    match args.model:
        case 'resnet':
            data = { split: ImageDataset(filepath=PROCESSED_DATA_PATH, split=split) for split in SPLITS }
            model = ResNet(data['train'].num_classes)
        case _:
            # default case
            data = { split: ImageDataset(filepath=PROCESSED_DATA_PATH, split=split) for split in SPLITS }
            model = ResNet(data['train'].num_classes)

    # initialise data loader
    loader = { split: DataLoader(data[split], batch_size=args.batch_size) for split in SPLITS}

    # define loss func, optimiser and lr scheduler
    criterion = nn.CrossEntropyLoss() # pyright: ignore
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, args.step_size, args.gamma)

    # train model
    start_task("Starting Training")
    print(get_summary(vars(args)))
    trained_model = train(model, loader['train'], loader['val'], criterion, optim, scheduler, args)

    # save model
    if args.save:
        filepath = os.path.join(MODEL_PATH, args.model)
        mkdir(filepath)
        save_path = os.path.join(filepath, str(datetime.datetime.now()))
        start_task(f"Saving Model to {save_path}")
        torch.save(trained_model.state_dict(), save_path)

        # log model to wandb
        artifact = wandb.Artifact(args.model, type="model")
        artifact.add_dir(filepath)
        wandb.log_artifact(artifact)

    # evaluate model
    if args.evaluate:
        start_task(f"Evaluating Model")
        labels = data['test'].labels
        id2label = data['test'].id2label
        y_true, y_pred, y_probs = get_predictions(trained_model, loader['test'], device=args.device)

        wandb.run.summary["Test Accuracy"] = np.mean(y_true == y_pred) # pyright: ignore

        # wandb visualisation
        conf_matrix = wandb.plot.confusion_matrix(None, list(y_true), list(y_pred), labels) # pyright: ignore
        roc_curve = wandb.plot.roc_curve(y_true, y_probs) # pyright: ignore
        pr_curve = wandb.plot.pr_curve(y_true, y_probs) # pyright: ignore

        # mispredicted images
        mispredictions = []
        for batch_num, (images, _) in enumerate(loader['test']):
            for i, image in enumerate(images):
                idx = batch_num * i + i
                true, pred = y_true[idx], y_pred[idx] # pyright: ignore
                if true != pred and len(mispredictions) < 20:
                    mispredictions.append(wandb.Image(unnormalise_image(image), caption=f"True: {id2label[true]} / Pred: {id2label[pred]}"))

        wandb.log({
            'Confusion Matrix': conf_matrix,
            'ROC Curve': roc_curve,
            'Precision-Recall Curve': pr_curve,
            'Mispredicted Images': mispredictions
        })

    end_task("Training Done", start_timer)

if __name__ == '__main__':
  main()
