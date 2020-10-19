import os
import cv2 as cv
import json
from glob import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt

def show_image(img):
    if img is None:
        print('Could not open or find the image: ', args.input)
        exit(0)
    else:
        '''cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
        plt.imshow(img, cmap='gray')
        plt.show()

def generate_mask(img):
    print("Reading Image:", img)
    image = cv.imread(img)
    #print(image)
    show_image(image)
    


if __name__ == "__main__":
     
    parser = argparse.ArgumentParser(description='Grayscale Mask Generation Script for segmentation of Gall Bladder Images')
    parser.add_argument('-i', '--img_path', type=str, default='img', required=True, help="Path for the image folder")
    parser.add_argument('-d', '--det_path', type=str, default='det', required=True, help="Path for the detected masks folder")
    
    args = parser.parse_args()
    
    img_path = args.img_path
    det_path = args.det_path
    images = glob(os.path.join(img_path,"*.jpg"))

    print("Total # images to process =", len(images))

    #for img in images:
    generate_mask(images[0])