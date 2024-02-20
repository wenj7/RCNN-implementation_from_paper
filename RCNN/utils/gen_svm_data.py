import os
import shutil
import cv2
import numpy as np
from util import check_dir, parse_car_csv, parse_xml, compute_ious

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

def get_neg_list(rects, rects_iou):
    neg_list = []
    for i in range(len(rects)):
        iou = rects_iou[i]
        rect = rects[i]
        if 0 < iou <= 0.3:
            neg_list.append(rect)
    return neg_list


if __name__ == '__main__':
    car_root_dir = './voc_data/voc_car/'
    svm_root_dir = './voc_data/svm_car/'
    check_dir(svm_root_dir)

    for dst_name in ['train', 'val']:
        dst_dir = os.path.join(car_root_dir, dst_name)

        # to copy images from orginal file to new file
        src_img_dir = os.path.join(svm_root_dir, dst_name, 'JPEGImages')
        check_dir(os.path.join(svm_root_dir, dst_name))
        check_dir(src_img_dir)
        dst_img_dir = os.path.join(car_root_dir, dst_name, 'JPEGImages')
        shutil.rmtree(src_img_dir, ignore_errors=True)
        shutil.copytree(dst_img_dir, src_img_dir)

        # to store src results: sample_name and its src results
        src_ann_dir = os.path.join(svm_root_dir, dst_name, 'Annotations')
        check_dir(os.path.join(svm_root_dir, dst_name))
        check_dir(src_ann_dir)
        shutil.rmtree(src_ann_dir, ignore_errors=True)
        check_dir(src_ann_dir)

        samples = parse_car_csv(dst_dir)
        imgs_path = [os.path.join(dst_dir,'JPEGImages',f'{sample}.jpg') for sample in samples]
        anns_path =[os.path.join(dst_dir,'Annotations',f'{sample}.xml') for sample in samples]

        for i in range(len(imgs_path)):
            sample = samples[i]
            img_path = imgs_path[i]
            ann_path = anns_path[i]
            img = cv2.imread(img_path)
            bboxs = parse_xml(ann_path)

            rects = gen_proposals_per_img(img)
            rects_iou = compute_ious(rects, bboxs)
            neg_list = get_neg_list(rects, rects_iou)
            pos_list = bboxs

            src_pos_path = os.path.join(src_ann_dir, sample+'_1.csv')
            src_neg_path = os.path.join(src_ann_dir, sample+'_0.csv')
                       
            np.savetxt(src_pos_path, np.array(pos_list), fmt='%d', delimiter=' ')
            np.savetxt(src_neg_path, np.array(neg_list), fmt='%d', delimiter=' ')
            print(f'{sample} already created its proposals and stored: pos-{len(pos_list)}, neg-{len(neg_list)}')