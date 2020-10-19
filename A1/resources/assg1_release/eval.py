#!/usr/bin/python3
import os
import cv2
import json
from glob import glob
import numpy as np
import argparse

def get_iou(det_mask, gt_mask):
    intersection = np.sum(cv2.bitwise_and(det_mask, gt_mask) > 0)
    union = np.sum(cv2.bitwise_or(det_mask, gt_mask) > 0)

    return intersection/union

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation Script for segmentation of Gall Bladder Images')
    parser.add_argument('-i', '--img_path', type=str, default='img', required=True, help="Path for the image folder")
    parser.add_argument('-d', '--det_path', type=str, default='det', required=True, help="Path for the detected masks folder")
    parser.add_argument('-g', '--gt_path', type=str, default='gt', required=True, help="Path for the ground truth masks folder")
    
    args = parser.parse_args()
    img_files = sorted(glob(os.path.join(args.img_path, "*jpg")))
    det_files = sorted(glob(os.path.join(args.det_path, "*")))
    gt_files = sorted(glob(os.path.join(args.gt_path, "*json")))
    print("Number of images: {}".format(len(img_files)))
    print("Number of detections: {}".format(len(det_files)))
    print("Number of ground truths: {}".format(len(gt_files)))

    assert(len(img_files) == len(det_files) == len(gt_files))

    iou = []
    for fimg, fdet, fgt in zip(img_files, det_files, gt_files):
        img = cv2.imread(fimg)

        with open(fgt, 'r') as f:
            gt = json.load(f)["shapes"][0]["points"]
        gt = np.array(gt, dtype=np.int32)

        gt_mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(gt_mask, [gt], [255, 255, 255])
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)

        det_mask = cv2.imread(fdet, cv2.CV_8UC1)

        assert(det_mask.shape == gt_mask.shape)
        iou.append(get_iou(det_mask, gt_mask))
        print("IoU for image {} = {}".format(fimg, iou[-1]))
    print("Average IoU = ", np.mean(iou))
