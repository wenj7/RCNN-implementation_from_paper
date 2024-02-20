import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
from tqdm import tqdm
from util import parse_car_csv, check_dir, parse_xml,compute_ious

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

def get_pos_neg_list(rects, rects_iou):
    pos_list = []
    neg_list = []
    for i in range(len(rects)):
        iou = rects_iou[i]
        rect = rects[i]
        x1,y1,x2,y2 = rect
        if x1!=x2 and y1!=y2:
            if iou>0.5:
                pos_list.append(rect)
            else:
                neg_list.append(rect)
    return pos_list, neg_list

if __name__ == '__main__':
    car_root_dir = './voc_data/voc_car/'
    cnnData_root_dir = './voc_data/cnn_car/'
    check_dir(cnnData_root_dir)

    for dst_name in ['train', 'val']:
        dst_dir = os.path.join(car_root_dir, dst_name)

        # to copy images from orginal file to new file
        src_img_dir = os.path.join(cnnData_root_dir, dst_name, 'JPEGImages')
        dst_img_dir = os.path.join(car_root_dir, dst_name, 'JPEGImages')
        shutil.rmtree(src_img_dir, ignore_errors=True)
        shutil.copytree(dst_img_dir, src_img_dir)

        # to store src results: sample_name and its src results
        src_ann_dir = os.path.join(cnnData_root_dir, dst_name, 'Annotations')
        shutil.rmtree(src_ann_dir, ignore_errors=True)
        check_dir(src_ann_dir)

        samples = parse_car_csv(dst_dir)
        imgs_path = [os.path.join(dst_dir,'JPEGImages',f'{sample}.jpg') for sample in samples]
        anns_path =[os.path.join(dst_dir,'Annotations',f'{sample}.xml') for sample in samples]
        for i in tqdm(range(len(imgs_path))):
            sample = samples[i]
            img_path = imgs_path[i]
            ann_path = anns_path[i]
            img = cv2.imread(img_path)
            bboxs = parse_xml(ann_path)

            rects = gen_proposals_per_img(img)
            rects_iou = compute_ious(rects, bboxs)
            pos_list, neg_list = get_pos_neg_list(rects, rects_iou)
            
            src_pos_path = os.path.join(src_ann_dir, sample+'_1.csv')
            src_neg_path = os.path.join(src_ann_dir, sample+'_0.csv')
                       
            np.savetxt(src_pos_path, np.array(pos_list), fmt='%d', delimiter=' ')
            np.savetxt(src_neg_path, np.array(neg_list), fmt='%d', delimiter=' ')
            # print(f'{sample} already created its proposals and stored: pos-{len(pos_list)}, neg-{len(neg_list)}')
        
        
            # img_show = img.copy()
            # for rect in rects:
            #     x1,y1,x2,y2 = rect
            #     cv2.rectangle(img_show,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255), 2)
            # plt.imshow(img_show)
            # plt.show()



