import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Sampler, DataLoader
from utils.custom_cnn_dataset import cnnDataset

class batchSampler(Sampler):
    def __init__(self, num_pos, num_neg, batch_pos, batch_neg):
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.batch_pos = batch_pos
        self.batch_neg = batch_neg
        self.batch = batch_pos + batch_neg  
        self.num_iter = (num_pos + num_neg) // self.batch
    
    def __iter__(self):
        sample_list = []
        idx_list = list(range(self.num_pos + self.num_neg))

        for i in range(self.num_iter):
            list1 = np.concatenate((random.sample(idx_list[:self.num_pos], self.batch_pos),
                        random.sample(idx_list[self.num_pos:], self.batch_neg)))
            random.shuffle(list1)
            sample_list.extend(list1)

        return iter(sample_list)
    
    def __len__(self):
        return self.num_iter * self.batch
    
def test():
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize((227, 227), antialias=True)
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = cnnDataset(root_dir = './voc_data/cnn_car/train', transform=transform)
    num_pos = dataset.total_pos_num
    num_neg = dataset.total_neg_num
    batch_pos = 32
    batch_neg = 32
    sampler = batchSampler(num_pos, num_neg, batch_pos, batch_neg)
    data_loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    for i, (X,y) in enumerate(data_loader):
        print(i)
        break

if __name__ == '__main__':
    test()