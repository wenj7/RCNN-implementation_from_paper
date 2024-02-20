import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from utils.custom_cnn_dataset import cnnDataset
from utils.batch_sampler import batchSampler
from utils.util import check_dir

def get_dataloader(dataset, batch_pos, batch_neg):
    num_pos = dataset.total_pos_num
    num_neg = dataset.total_neg_num
    
    sampler = batchSampler(num_pos, num_neg, batch_pos, batch_neg)
    data_loader = DataLoader(dataset, batch_size=batch_pos+batch_neg, sampler=sampler)

    return data_loader

def train(model, device, criterion, optimizer, transform, batch_pos, batch_neg, num_epochs, lr_scheduler):
    
    if device is not None:
        model = model.to(device)
        criterion = criterion.to(device)
    
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0
    
    for epoch in range(num_epochs):
        # 1. train the model 
        model.train()
        train_dataset = cnnDataset(root_dir = './voc_data/cnn_car/train', transform=transform)
        train_data_loader = get_dataloader(train_dataset, batch_pos, batch_neg)
        batch_loss = 0
        batch_corrects = 0
        val_batch_loss = 0
        val_batch_corrects = 0
        
        for i, (X,y) in enumerate(train_data_loader):
            if device is not None:
                X = X.to(device)
                y = y.to(device)
            y_hat = model(X)
            y = y.to(dtype=torch.long)
            loss = criterion(y_hat, y)
            _, preds = torch.max(y_hat, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss += loss.item()
            batch_corrects += torch.sum(preds == y)
            print(f'{epoch} epoch, {i} batch: loss {loss.item()}, accuracy {torch.sum(preds == y)/X.size(0)}')

            lr_scheduler.step()

        epoch_loss = batch_loss/train_data_loader.__len__()
        epoch_accuracy = batch_corrects/(train_data_loader.__len__() * X.size(0))
        print(f'{epoch} epoch: loss is {epoch_loss}, accuracy is {epoch_accuracy}')

        # 2. val the trained model
        model.eval()
        val_dataset = cnnDataset(root_dir = './voc_data/cnn_car/val', transform=transform)
        val_data_loader = get_dataloader(val_dataset, batch_pos, batch_neg)
        for i, (X,y) in enumerate(val_data_loader):
            if device is not None:
                X = X.to(device)
                y = y.to(device)
            y_hat = model(X)
            y = y.to(dtype=torch.long)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            val_batch_loss += loss.item()
            val_batch_corrects += torch.sum(preds == y)
        val_epoch_loss = val_batch_loss/val_data_loader.__len__()
        val_epoch_accuracy = val_batch_corrects/(val_data_loader.__len__()* X.size(0))
        print(f'{epoch} epoch: loss is {val_epoch_loss}, accuracy is {val_epoch_accuracy}')

        if val_epoch_accuracy > best_acc:
            best_acc = val_epoch_accuracy
            best_model_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_weights)

    return model

if __name__ == '__main__':
    # 1. select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print('device is available')
    # device = None
    # 2. select model, optimizer, loss, transform, and batch_size
    model = models.alexnet(pretrained= True)
    # features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(4096, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((227, 227), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_pos = 32
    batch_neg = 96
    num_epochs = 2

    # 4. train the model
    best_model = train(model, device, criterion, optimizer, transform, batch_pos, batch_neg, num_epochs, lr_scheduler)
    check_dir('./voc_data/models')
    torch.save(best_model.state_dict(), './voc_data/models/alexnet_car.pth')
