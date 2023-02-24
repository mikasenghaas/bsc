# train.py
#  by: mika senghaas

import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from data import ImageDataset
from model import ResNet
from utils import *

def train(model, train_loader, val_loader, criterion, optim, scheduler, max_epochs, device):
    model.to(device)
    pbar = tqdm(range(max_epochs))
    pbar.set_description(f'XXX/XX - Train: X.XXX (XX.X%) - Val: X.XXX (XX.X%)')
    train_loss, val_loss = 0.0, 0.0
    train_acc, val_acc = 0.0, 0.0
    for epoch in pbar:
        running_loss, running_correct = 0.0, 0
    
        model.train()
        for batch_num, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
  
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
            samples_seen = (batch_num + 1) * BATCH_SIZE

            # normalise
            train_acc = running_correct / samples_seen
            train_loss = running_loss / samples_seen
            
            pbar.set_description(f'{str(epoch).zfill(2)}/{str(batch_num).zfill(3)} - Train: {train_loss:.3f} ({(train_acc * 100):.1f}%) - Val: {val_loss:.3f} ({(val_acc * 100):.1f}%)')

        if val_loader != None:
            running_loss, running_correct = 0.0, 0
            model.eval()
            for batch_num, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
      
                logits = model(inputs)
                preds = torch.argmax(logits, 1)
                loss = criterion(logits, labels)

                # accumulate loss and correct predictions
                running_loss += loss.item()
                running_correct += torch.sum(labels == preds)

            val_loss = running_loss / len(val_loader.dataset)
            val_acc = running_correct / len(val_loader.dataset)
            pbar.set_description(f'{str(epoch).zfill(2)}/{str(batch_num).zfill(3)} - Train: {train_loss:.3f} ({(train_acc * 100):.1f}%) - Val: {val_loss:.3f} ({(val_acc * 100):.1f}%)')

    return model

def main():
    parser = load_train_parser()
    args = parser.parse_args()

    start_timer = start_task("Running train.py", get_timer=True)

    # load data 
    start_task("Initialising Data and Model")
    match args.model:
        case 'resnet':
            train_data = ImageDataset(filepath=PROCESSED_DATA_PATH, split="train")
            val_data = ImageDataset(filepath=PROCESSED_DATA_PATH, split="val")
            model = ResNet(train_data.num_classes)
        case _:
            # default case
            train_data = ImageDataset(filepath=PROCESSED_DATA_PATH, split="train")
            val_data = ImageDataset(filepath=PROCESSED_DATA_PATH, split="val")
            model = ResNet(train_data.num_classes)

    # initialise data loader
    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    # define loss func, optimiser and lr scheduler
    criterion = nn.CrossEntropyLoss() # pyright: ignore
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, args.step_size, args.gamma)

    # train model
    start_task("Starting Training")
    print(get_summary(vars(args)))
    trained_model = train(model, train_loader, val_loader, criterion, optim, scheduler, args.max_epochs, args.device)

    # save model
    if args.save:
        filepath = os.path.join(MODEL_PATH, args.model)
        mkdir(filepath)
        save_path = os.path.join(filepath, str(datetime.datetime.now()))
        start_task(f"Saving Model to {save_path}")
        torch.save(trained_model.state_dict(), save_path)

    end_task("Training Done", start_timer)

if __name__ == '__main__':
  main()
