import cv2
import copy
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from utils.util import parse_xml
from torchvision.models import alexnet
from utils.util import iou
from box_regressor import xyxy_to_xyhw

def gen_proposals_per_img(img):
    cv2.setUseOptimized(True)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    # from (x,y,w,h) to (x1,y1,x2,y2)
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]

    return rects

def get_model(device=None):
    model = alexnet()
    num_classes = 2
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load('./voc_data/models/svm_car.pth'))
    model.eval()

    # 取消梯度追踪
    for param in model.parameters():
        param.requires_grad = False
    if device:
        model = model.to(device)

    return model

def get_iou(pred_box, target_box):
    xA = np.maximum(pred_box[0], target_box[0])
    yA = np.maximum(pred_box[1], target_box[1])
    xB = np.minimum(pred_box[2], target_box[2])
    yB = np.minimum(pred_box[3], target_box[3])
    # 计算交集面积
    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    # 计算两个边界框面积
    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])

    scores = intersection / (boxAArea + boxBArea - intersection)
    return scores

def nms(positive_list, score_list):
    # begin
    D = {}
    B = {}
    S = {}
    
    nms_thresh = 0.3
    for idx, rect in enumerate(positive_list):
        B[idx] = rect
    for idx, score in enumerate(score_list):
        S[idx] = score
    m=0
    S2 = S.copy()
    # while B is not empty
    while B:
        for key, value in S2.items():
            if key not in D.keys() and value == max(S2.values()):
                M = B[key]
                D[key] = M
                B.pop(key)
                S2.pop(key)
                break

        copy_dict = B.copy()
        for idx, box in copy_dict.items():
            if get_iou(M, box)>=nms_thresh:
                B.pop(idx)
                S.pop(idx)
                S2.pop(idx)
    return D, S

def draw_results(img, rect_list, score_list, box_reg):
    for key, value in rect_list.items():
        xmin, ymin, xmax, ymax = value
        score = score_list[key]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=1)
        cv2.putText(img, "{:.3f}".format(score), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for rect in box_reg:
        xmin, ymin, xmax, ymax = rect
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 255, 0), thickness=1)
    cv2.imshow("img",img)
    cv2.waitKey(0)

def bbox_regression(img, nms_rects, w_hat):
    # use for convlve
    device = torch.device("mps")
    model = models.alexnet()
    model.classifier[6] = nn.Linear(4096, 2)
    model_path = 'voc_data/models/alexnet_car.pth'
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((227, 227), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    bbox_reg = []
    print('norm of w_hat:', np.linalg.norm(w_hat))
    for key, value in nms_rects.items():
        xmin,ymin,xmax,ymax = value

        proposal = img[xmin:xmax,ymin:ymax]
        if 0 not in proposal.shape:
            proposal = transform(proposal)
            proposal = proposal.to(device)

            px,py,pw,ph = xyxy_to_xyhw(value,single=1)
            print('original proposal:',px,py,pw,ph)
            output = model.features(proposal).cpu().detach().numpy().reshape(-1)
            
            Gx = pw*w_hat[0,:].dot(output)+px
            Gy = ph*w_hat[1,:].dot(output)+py
            Gw = pw*np.exp(w_hat[2,:].dot(output))
            Gh = ph*np.exp(w_hat[3,:].dot(output))
            print('bbox regression:',Gx,Gy,Gw,Gh)
            bbox_reg.append([Gx,Gy,Gx+Gw,Gy+Gh])
    return bbox_reg

if __name__ == '__main__':
    device = torch.device("mps")
    model = get_model(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((227, 227), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_img_path = './voc_data/voc_car/val/JPEGImages/007422.jpg'
    test_xml_path = './voc_data/voc_car/val/Annotations/007422.xml'
    img = cv2.imread(test_img_path)
    dst = copy.deepcopy(img)

    bboxs = parse_xml(test_xml_path)

    proposals = gen_proposals_per_img(img)
    print('候选区域建议数目： %d' % len(proposals))

    svm_thresh = 0.6

    score_list = []
    positive_list = []

    
    for rect in proposals :
        xmin, ymin, xmax, ymax = rect
        rect_img = img[ymin:ymax, xmin:xmax]

        rect_transform = transform(rect_img).to(device)
        output = model(rect_transform.unsqueeze(0))[0]

        if torch.argmax(output).item() == 1:
            """
            预测为汽车
            """
            probs = torch.softmax(output, dim=0).cpu().numpy()

            if probs[1] >= svm_thresh:
                score_list.append(probs[1])
                positive_list.append(rect)


    nms_rects, nms_scores = nms(positive_list, score_list)

    w_hat = np.load('./voc_data/w_hat.npy')
    box_reg = bbox_regression(dst, nms_rects, w_hat)

    print(nms_rects, box_reg, '\n', nms_scores)

    draw_results(dst, nms_rects, nms_scores, box_reg)
    
