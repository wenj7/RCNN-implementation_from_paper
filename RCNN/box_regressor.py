import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import utils.util as util

def get_bndbox(bndboxes, positive):
        """
        返回和positive的IoU最大的标注边界框
        :param bndboxes: 大小为[N, 4]或者[4]
        :param positive: 大小为[4]
        :return: [4]
        """

        if len(bndboxes.shape) == 1:
            # 只有一个标注边界框，直接返回即可
            return bndboxes
        else:
            scores = util.iou(positive, bndboxes)
            return bndboxes[np.argmax(scores)]

def load_data(data_dir):
    # 1. load gt box and proposals
    samples = util.parse_car_csv(data_dir)
    gt_list = []
    p_list = []
    img_list = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((227, 227), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    gt_annotation_dir = os.path.join(data_dir,'bndboxs')
    proposals_dir = os.path.join(data_dir,'positive')
    images_dir = os.path.join(data_dir,'JPEGImages')

    for i in range(len(samples)):
        sample_name = samples[i]

        jpeg_path = os.path.join(images_dir, sample_name + '.jpg')
        bndbox_path = os.path.join(gt_annotation_dir, sample_name + '.csv')
        proposals_path = os.path.join(proposals_dir, sample_name + '.csv')

        image = cv2.imread(jpeg_path)
        bndboxes = np.loadtxt(bndbox_path, dtype=int, delimiter=' ')
        positives = np.loadtxt(proposals_path, dtype=int, delimiter=' ')

        if len(positives.shape) == 1:
            bndbox = get_bndbox(bndboxes, positives)
            gt_list.append(bndbox)
            p_list.append(positives)
            xmin,ymin,xmax,ymax = bndbox
            img = image[ymin:ymax, xmin:xmax]
            img = transform(img)
            img_list.append(img)
        else:
            for positive in positives:
                bndbox = get_bndbox(bndboxes, positive)
                gt_list.append(bndbox)
                p_list.append(positive)
                xmin,ymin,xmax,ymax = bndbox
                img = image[ymin:ymax, xmin:xmax]
                img = transform(img)
                img_list.append(img)
        
    return img_list, gt_list, p_list

def xyxy_to_xyhw(box_list, single = 0):
    xyhw = []
    if not single:
        for box in box_list:
            gx,gy,x2,y2 = box
            gw,gh = x2-gx, y2-gy
            xyhw.append([gx,gy,gw,gh])
        xyhw = np.array(xyhw)
        return xyhw
    else:
        gx,gy,x2,y2 = box_list
        gw,gh = x2-gx, y2-gy
        return (gx,gy,gw,gh)
    

def least_square_solve(gt_list, p_list, features, alpha):
    gt_mat = xyxy_to_xyhw(gt_list)
    p_mat = xyxy_to_xyhw(p_list)
    tx = (gt_mat[:,0] - p_mat[:,0])/p_mat[:,2]
    ty = (gt_mat[:,1] - p_mat[:,1])/p_mat[:,3]
    tw = np.log(gt_mat[:,2]/p_mat[:,2])
    th = np.log(gt_mat[:,3]/p_mat[:,3])
    w_hat = []
    features = np.array(features)
    X_b = features.reshape(features.shape[0], -1)
    for y in [tx,ty,tw,th]:
        I_mat = np.identity(X_b.shape[1])    
        optimal_w = np.linalg.inv(X_b.T.dot(X_b) + alpha * I_mat).dot(X_b.T).dot(y)
        w_hat.append(optimal_w)
    
    w_hat = np.array(w_hat)
    return w_hat

def estimate_w_hat(alpha):
    # 1. 加载已经训练好的cnn模型, 用于提取pool5层特征
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model = models.alexnet()
    model.classifier[6] = nn.Linear(4096, 2)
    model_path = 'voc_data/models/alexnet_car.pth'
    model.load_state_dict(torch.load(model_path))

    # 2. 加载用于bbox regression的数据
    bbox_regression_data_dir = './voc_data/bbox_regression'
    img_list, gt_list, p_list = load_data(bbox_regression_data_dir)

    # 3. 提取pool5的输出
    model.eval()
    if device is not None:
        model = model.to(device)
    
    pool5_output = []
    for i in range(len(img_list)):
        img = img_list[i]
        img = img.to(device)
        pool5_output.append(model.features(img).cpu().detach().numpy())

    w_hat = least_square_solve(gt_list, p_list, pool5_output, alpha)

    return w_hat    

def test(w_hat):
    device = torch.device("mps")
    model = models.alexnet()
    model.classifier[6] = nn.Linear(4096, 2)
    model_path = 'voc_data/models/alexnet_car.pth'
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    bbox_regression_data_dir = './voc_data/bbox_regression'
    img_list, gt_list, p_list = load_data(bbox_regression_data_dir)
    proposal = img_list[0]
    proposal = proposal.to(device)
    px,py,pw,ph = xyxy_to_xyhw(p_list[0],single=1)
    
    output = model.features(proposal).cpu().detach().numpy().reshape(-1)
    Gx = pw*w_hat[0,:].dot(output)+px
    Gy = ph*w_hat[1,:].dot(output)+py
    Gw = pw*np.exp(w_hat[2,:].dot(output))
    Gh = ph*np.exp(w_hat[3,:].dot(output))
    
    print([px,py,pw,ph], [Gx,Gy,Gw,Gh], xyxy_to_xyhw(gt_list[0],single=1))


if __name__ == '__main__':

    w_hat = estimate_w_hat(alpha=10)
    print(np.linalg.norm(w_hat))
    np.save('./voc_data/w_hat.npy', w_hat)
    a = np.load('./voc_data/w_hat.npy')
    print(a)
    print(a.shape)
    # test(w_hat)
    
    
