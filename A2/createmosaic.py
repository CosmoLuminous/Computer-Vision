'''
*********************************************
# TITLE: ASSIGNMENT 2:IMAGE MOSAIC CREATION # 
#        Image Stitching                    #
# COURSE: COL780 - COMPUTER VISION          #
# INSTRUCTOR: PROF. CHETAN ARORA            #
# AUTHOR: AMAN BHARDWAJ                     #
# DATE: 20 NOV 2020                         #
*********************************************
'''
''' IMPORT PACKAGES '''
import os
import cv2 as cv
import json
from glob import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
     
    parser = argparse.ArgumentParser(description='Grayscale Mask Generation Script for segmentation of Gall Bladder Images')
    parser.add_argument('-i', '--input_path', type=str, default='./val_set/1', required=True, help="Path for the image folder")
    args = parser.parse_args()
    input_path = args.input_path
    print(input_path)

