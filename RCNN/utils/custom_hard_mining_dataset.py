# -*- coding: utf-8 -*-

"""
refer from:
@file: custom_hard_negative_mining_dataset.py
@author: zj
@description: 
"""

import torch.nn as nn
from torch.utils.data import Dataset
from utils.custom_svm_dataset import svmDataset


class hnmDataset(Dataset):

    def __init__(self, negative_list, jpeg_images, transform=None):
        self.negative_list = negative_list
        self.jpeg_images = jpeg_images
        self.transform = transform

    def __getitem__(self, index: int):
        target = 0

        negative_dict = self.negative_list[index]
        xmin, ymin, xmax, ymax = negative_dict['rect']
        image_id = negative_dict['image_id']

        image = self.jpeg_images[image_id][ymin:ymax, xmin:xmax]
        if self.transform:
            image = self.transform(image)

        return image, target, negative_dict

    def __len__(self) -> int:
        return len(self.negative_list)


if __name__ == '__main__':
    root_dir = 'voc_data/cnn_car/train'
    data_set = svmDataset(root_dir)

    negative_list = data_set.get_negatives()
    jpeg_images = data_set.get_jpeg_images()
    transform = data_set.get_transform()

    hard_negative_dataset = hnmDataset(negative_list, jpeg_images, transform=transform)
    image, target, negative_dict = hard_negative_dataset.__getitem__(100)

    print(image.shape)
    print(target)
    print(negative_dict)