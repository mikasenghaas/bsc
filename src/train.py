# train.py
#  by: mika senghaas

from timeit import default_timer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from config import *
from data import ImageDataset
from transform import ImageTransformer
from model import FinetunedImageClassifier
from utils import *

def train(model, transform, train_loader, val_loader, criterion, optim, scheduler, args):
    model.to(args.device)
    pbar = tqdm(range(args.max_epochs))
    pbar.set_description(f'XXX/XX (XX.Xms/ XX.Xms) - Train: X.XXX (XX.X%) - Val: X.XXX (XX.X%)')
    train_loss, val_loss = 0.0, 0.0
    train_acc, val_acc = 0.0, 0.0
    training_times, inference_times = [], []
    for epoch in pbar:
        running_loss, running_correct = 0.0, 0
        running_training_time, running_inference_time = 0.0, 0.0
        model.train()
        for batch_num, (inputs, labels) in enumerate(train_loader):
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
            running_correct += torch.sum(preds == labels)
            samples_seen = (batch_num + 1) * args.batch_size

            # normalise
            train_acc = running_correct / samples_seen
            train_loss = running_loss / samples_seen
            
            pbar.set_description(f'{str(epoch).zfill(len(str(args.max_epochs)))}/{str(batch_num).zfill(len(str(len(train_loader))))} ({round(running_training_time / samples_seen * 1000, 1)}ms | {round(running_inference_time / samples_seen * 1000, 1)}ms) - Train: {train_loss:.3f} ({(train_acc * 100):.1f}%) - Val: {val_loss:.3f} ({(val_acc * 100):.1f}%)')

            # log epoch metrics for train and val split
            if args.log:
                wandb.log({
                    'training_accuracy': train_acc, 
                    'validation_accuracy': val_acc,
                    'training_loss': train_loss, 
                    'validation_loss': val_loss})

        
        training_times.append(running_training_time)
        inference_times.append(running_inference_time)
                
        if val_loader != None:
            running_loss, running_correct = 0.0, 0
            model.eval()
            for batch_num, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
      
                logits = model(transform(inputs))
                preds = torch.argmax(logits, 1)
                loss = criterion(logits, labels)

                # accumulate loss and correct predictions
                running_loss += loss.item()
                running_correct += torch.sum(labels == preds)

            val_loss = running_loss / len(val_loader.dataset)
            val_acc = running_correct / len(val_loader.dataset)

            pbar.set_description(f'{str(epoch).zfill(len(str(args.max_epochs)))}/00 - Train: {train_loss:.3f} ({(train_acc * 100):.1f}%) - Val: {val_loss:.3f} ({(val_acc * 100):.1f}%)')

        # adjust learning rate
        scheduler.step()

    # log average training step time/ sample + inference time/ sample
    if args.log:
        wandb.config.update({
            "training_time_per_sample_ms" : round(sum(training_times) / len(training_times), 1),
            "inference_time_per_sample_ms" : round(sum(inference_times) / len(inference_times), 1)
            })

    return model

def main():
    start_timer = start_task("Running train.py", get_timer=True)

    # parse cli arguments
    args = load_train_args()

    # initialise wandb run
    if args.log:
        wandb.init(project="bsc", name=args.model, notes="computer vision models for indoor localisation", config=vars(args))

        wandb.define_metric("training_loss", summary="min")
        wandb.define_metric("validation_loss", summary="min")
        wandb.define_metric("training_accuracy", summary="max")
        wandb.define_metric("validation_accuracy", summary="max")

    # load data 
    start_task("Initialising Data and Model")

    # initialise data, tranforms and model
    data = { split: ImageDataset(filepath=PROCESSED_DATA_PATH, split=split) for split in SPLITS }
    transform = ImageTransformer()
    model = FinetunedImageClassifier(args.model, pretrained=args.pretrained, id2label=data['train'].id2label)

    # initialise data loader
    loader = { split: DataLoader(data[split], batch_size=args.batch_size) for split in SPLITS}

    # define loss func, optimiser and lr scheduler
    criterion = nn.CrossEntropyLoss() # pyright: ignore
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, args.step_size, args.gamma)

    # train model
    start_task("Starting Training")
    print(get_summary(vars(args)))
    trained_model = train(model, transform, loader['train'], loader['val'], criterion, optim, scheduler, args)

    if args.log:
        # log meta information
        wandb.config.update({"num_params": model.meta['num_params']})
        wandb.summary["dataset"] = data['train'].meta 
        wandb.summary["model"] = model.meta

        # prepare artifact saving
        filepath = os.path.join(MODEL_PATH, args.model)
        mkdir(filepath)

        # save transforms
        start_task(f"Saving Transforms and Model to {filepath}")
        save_pickle(transform, os.path.join(filepath, f"transforms.pkl"))
        torch.save(trained_model.state_dict(), os.path.join(filepath, f"{args.model}.pt"))

        # save as artifact to wandb
        start_task("Saving Artifcats to WANDB")
        artifact = wandb.Artifact(args.model, type="model")
        artifact.add_dir(filepath)
        wandb.log_artifact(artifact)

        start_task(f"Evaluating Model")
        labels = data['test'].labels
        id2label = data['test'].id2label
        y_true, y_pred, y_probs = get_predictions(trained_model, transform, loader['test'], device=args.device)

        wandb.run.summary["test_accuracy"] = np.mean(y_true == y_pred) # pyright: ignore

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
            'confusion_matrix': conf_matrix,
            'roc_curve': roc_curve,
            'pr_curve': pr_curve,
            'mispredictions': mispredictions
        })

    end_task("Training Done", start_timer)

if __name__ == '__main__':
  main()
