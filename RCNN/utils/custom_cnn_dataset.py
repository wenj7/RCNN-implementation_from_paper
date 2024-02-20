import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
from torch.utils.data import Dataset
from utils.util import parse_car_csv

class cnnDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        samples = parse_car_csv(root_dir)

        pos_anns_path = [os.path.join(root_dir, 'Annotations', sample_name + '_1.csv') for sample_name in samples]
        neg_anns_path = [os.path.join(root_dir, 'Annotations', sample_name + '_0.csv') for sample_name in samples]
        self.imgs_path = [os.path.join(root_dir, 'JPEGImages', sample_name + '.jpg') for sample_name in samples]

        pos_rects = []
        neg_rects = []

        pos_sizes = []
        neg_sizes = []

        for path in pos_anns_path:
            rects = np.loadtxt(path, dtype=int, delimiter=' ')
            # check samples whose rects are empty or only contain one row
            if len(rects.shape) == 1:
                # contaion only one row
                if rects.shape[0] == 4:
                    pos_rects.append(rects)
                    pos_sizes.append(1)
                # empty
                else:
                    pos_sizes.append(0)
            else:
                pos_rects.extend(rects)
                pos_sizes.append(len(rects))

        for path in neg_anns_path:
            rects = np.loadtxt(path, dtype=int, delimiter=' ')
            if len(rects.shape) == 1:
                if rects.shape[0] == 4:
                    neg_rects.append(rects)
                    neg_sizes.append(1)
                else:
                    neg_sizes.append(0)
            else:
                neg_rects.extend(rects)
                neg_sizes.append(len(rects))

        self.total_pos_num = sum(pos_sizes)
        self.total_neg_num = sum(neg_sizes)
        self.pos_sizes = pos_sizes
        self.neg_sizes = neg_sizes
        self.pos_rects = pos_rects
        self.neg_rects = neg_rects
        self.transform = transform

    def __getitem__(self, index):
        # assignment of the final index value to avoid error
        image_index = len(self.imgs_path)-1
        if index < self.total_pos_num:
            label = 1
            # search for img_id
            x1,y1,x2,y2 = self.pos_rects[index]
            for i in range(len(self.pos_sizes)-1):
                if sum(self.pos_sizes[:i]) <= index < sum(self.pos_sizes[:i+1]):
                    image_index = i
                    break

        else:
            label = 0
            index = index - self.total_pos_num
            x1,y1,x2,y2 = self.neg_rects[index]
            for j in range(len(self.neg_sizes)-1):
                if sum(self.neg_sizes[:j]) <= index < sum(self.neg_sizes[:j+1]):
                    image_index = j
                    break
        img = cv2.imread(self.imgs_path[image_index])[y1:y2, x1:x2]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return self.total_pos_num + self.total_neg_num

    
if __name__ == '__main__':
    
    # orgin_file = './voc_data/voc_car/val/car.csv'
    # des_file = './voc_data/cnn_car/val/car.csv'
    # shutil.copy2(orgin_file,des_file)

    root_dir = './voc_data/cnn_car/train'
    # dataset = cnnDataset(root_dir)
    # img, label = dataset[2]
    # plt.imshow(img)
    # plt.show()
    