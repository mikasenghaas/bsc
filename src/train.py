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

def train(model, train_loader, criterion, optim, scheduler, max_epochs, val_loader=None):
    model.to(DEVICE)
    model.train()
    pbar = tqdm(range(max_epochs))
    pbar.set_description(f'EPOCH X - BATCH X - LOSS X.XXX - ACC XX.X%')
    for epoch in pbar:
        running_loss = 0.0
        running_correct = 0
    
        for batch_num, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
  
            # zero the parameter gradients
            optim.zero_grad()
  
            logits = model(inputs)
            _, preds = torch.max(logits, 1)
            loss = criterion(logits, labels)
  
            # backward + optimize only if in training phase
            loss.backward()
            optim.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(labels == preds)
  
            samples_seen = (batch_num + 1) * BATCH_SIZE
            pbar.set_description(f'EPOCH {epoch} - BATCH {batch_num} - LOSS {(running_loss / samples_seen):.3f} - ACC {(running_correct / samples_seen * 100):.1f}%')

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
            model = ResNet(train_data.num_classes)
        case _:
            # default case
            train_data = ImageDataset(filepath=PROCESSED_DATA_PATH, split="train")
            model = ResNet(train_data.num_classes)

    # initialise data loader
    train_loader = DataLoader(train_data, batch_size=args.batch_size)

    # define loss func, optimiser and lr scheduler
    criterion = nn.CrossEntropyLoss() # pyright: ignore
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, args.step_size, args.gamma)

    # train model
    start_task("Starting Training")
    print(get_summary(vars(args)))
    trained_model = train(model, train_loader, criterion, optim, scheduler, args.max_epochs)

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
