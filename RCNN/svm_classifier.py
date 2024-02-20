import os
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import alexnet

from utils.util import save_model
from utils.custom_svm_dataset import svmDataset
from utils.batch_sampler import batchSampler
from utils.custom_hard_mining_dataset import hnmDataset

def get_dataloader(data_root_dir, phase, batch_positive, batch_negative):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((227, 227), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_dir = os.path.join(data_root_dir, phase)
    data_set = svmDataset(data_dir, transform=transform)

    # in train stage, we need to record the negative list, which will be used to hard negative mining
    if phase == 'train':
        data_size = 0
        positive_list = data_set.get_positives()
        negative_list = data_set.get_negatives()

        init_negative_idxs = random.sample(range(len(negative_list)), len(positive_list))
        init_negative_list = [negative_list[idx] for idx in range(len(negative_list)) if idx in init_negative_idxs]
        remain_negative_list = [negative_list[idx] for idx in range(len(negative_list))
                                    if idx not in init_negative_idxs]
        data_set.set_negative_list(init_negative_list)
    
        sampler = batchSampler(data_set.get_positive_num(), data_set.get_negative_num(),
                                     batch_positive, batch_negative)

        data_loader = DataLoader(data_set, batch_size=batch_positive+batch_negative, sampler=sampler, num_workers=8, drop_last=True)
        data_size = len(sampler)
        return data_set, data_loader, remain_negative_list, data_size
    # in valid stage, hard negative mining is not involved
    else:
        sampler = batchSampler(data_set.get_positive_num(), data_set.get_negative_num(),
                                     batch_positive, batch_negative)
        data_loader = DataLoader(data_set, batch_size=batch_positive+batch_negative, sampler=sampler, num_workers=8, drop_last=True)
        data_size = len(sampler)
        return data_set, data_loader, data_size

def add_hard_negatives(hard_negative_list, negative_list, add_negative_list):
     for item in hard_negative_list:
        if len(add_negative_list) == 0:
            # 第一次添加负样本
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))
        if list(item['rect']) not in add_negative_list:
            negative_list.append(item)
            add_negative_list.append(list(item['rect']))


def get_hard_negatives(preds, cache_dicts):
    fp_mask = preds == 1
    tn_mask = preds == 0

    fp_rects = cache_dicts['rect'][fp_mask].numpy()
    fp_image_ids = cache_dicts['image_id'][fp_mask].numpy()

    tn_rects = cache_dicts['rect'][tn_mask].numpy()
    tn_image_ids = cache_dicts['image_id'][tn_mask].numpy()

    hard_negative_list = [{'rect': fp_rects[idx], 'image_id': fp_image_ids[idx]} for idx in range(len(fp_rects))]
    easy_negatie_list = [{'rect': tn_rects[idx], 'image_id': tn_image_ids[idx]} for idx in range(len(tn_rects))]

    return hard_negative_list, easy_negatie_list


def train(data_path, model, criterion, optimizer, lr_scheduler, num_epochs, device, batch_positive, batch_negative):
    model = model.to(device)
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_dataset, train_dataloader, remain_negative_list, train_data_size = get_dataloader(data_path, 'train', batch_positive, batch_negative)
    val_dataset, val_dataloader, val_data_size = get_dataloader(data_path, 'val', batch_positive, batch_negative)
    add_negative_list = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 1. train
        model.train()
        running_loss = 0.0
        running_corrects = 0

        print(f'train - positive_num: {train_dataset.get_positive_num()} - negative_num: {train_dataset.get_negative_num()} - data size: {train_data_size}')
        
        for X, y, target in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                y_hat = model(X)
                _, preds = torch.max(y_hat, 1)
                outputs = y_hat[range(len(y)),y]
                # print(outputs)
                loss = criterion(outputs, y, margin=1.0)
                
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * X.size(0)
            running_corrects += torch.sum(preds == y.data)
        lr_scheduler.step()
        epoch_loss = running_loss / train_data_size
        epoch_acc = running_corrects.float() / train_data_size

        print(f'train Loss: {epoch_loss} Acc: {epoch_acc}' )


        # 2. hard negative mining
        jpeg_images = train_dataset.get_jpeg_images()
        transform = train_dataset.get_transform()
        with torch.set_grad_enabled(False):
            remain_dataset = hnmDataset(remain_negative_list, jpeg_images, transform=transform)
            remain_data_loader = DataLoader(remain_dataset, batch_size=batch_positive+batch_negative, num_workers=8, drop_last=True)

            # 获取训练数据集的负样本集
            negative_list = train_dataset.get_negatives()
            # 记录后续增加的负样本
            if add_negative_list:
                add_negative_list

            running_corrects = 0
            # Iterate over data.
            for inputs, labels, cache_dicts in remain_data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)
                # print(outputs.shape)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels.data)

                hard_negative_list, easy_neagtive_list = get_hard_negatives(preds.cpu().numpy(), cache_dicts)
                add_hard_negatives(hard_negative_list, negative_list, add_negative_list)

            remain_acc = running_corrects.float() / len(remain_negative_list)
            print('remiam negative size: {}, acc: {:.4f}'.format(len(remain_negative_list), remain_acc))

            # 训练完成后，重置负样本，进行hard negatives mining
            train_dataset.set_negative_list(negative_list)
            tmp_sampler = batchSampler(train_dataset.get_positive_num(), train_dataset.get_negative_num(),
                                             batch_positive, batch_negative)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_positive+batch_negative, sampler=tmp_sampler,
                                               num_workers=8, drop_last=True)
            # 重置数据集大小
            train_data_size = len(tmp_sampler)


        # 3. valid
        model.eval()
    
        print(f'val - positive_num: {val_dataset.get_positive_num()} - negative_num: {val_dataset.get_negative_num()} - data size: {val_data_size}')
        running_loss = 0.0
        running_corrects = 0
        for X, y, target in val_dataloader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            outputs = y_hat[range(len(y)),y]
            _, preds = torch.max(y_hat, 1)

            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
            running_corrects += torch.sum(preds == y.data)

        epoch_loss = running_loss / val_data_size
        epoch_acc = running_corrects.float() / val_data_size
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_weights = copy.deepcopy(model.state_dict())
        print(f'val Loss: {epoch_loss} Acc: {epoch_acc}' )

    model.load_state_dict(best_model_weights)
    return model

if __name__ == '__main__':
    # 1. device and data
    device = torch.device("mps" if torch.backends.mps.is_available() else 'cpu')
    data_path = 'voc_data/svm_car'
    # 2. model setting
    model_path = 'voc_data/models/alexnet_car.pth'
    model = alexnet()
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Linear(num_features, 2)
    # 3. train setting
    criterion = F.hinge_embedding_loss
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    batch_positive = 32
    batch_negative = 96
    num_epochs = 10
    # batch_total = 128

    # 4. get the best model
    best_model = train(data_path, model, criterion, optimizer, lr_scheduler, 
                       num_epochs, device, batch_positive, batch_negative)
    # 5. store weights
    save_model(best_model, 'voc_data/models/svm_car.pth')