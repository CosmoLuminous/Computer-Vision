import os
import cv2 as cv
import json
from glob import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt


''' DEFINE GLOBAL VARIABLES '''
RECT_KERNEL_5X5 = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
RECT_KERNEL_3X3 = cv.getStructuringElement(cv.MORPH_RECT,(3,3))


''' UTIL FUNCTIONS '''
def hconcat_images(image_list):
    return cv.hconcat(image_list)

def show_image(img, title='Default', save=False):
    if img is None:
        print('Could not find image to show.')
    else:        
        print("\n\n%%%% IMAGE: {}, SHAPE: {} %%%%".format(title, img.shape))
        fig = plt.figure(0)
        plt.imshow(img, cmap='gray')
        plt.show()
        if save:
            fig.savefig("plots/"+name+".png", dpi=300, format="png")        
    return

def save_mask(mask, name):
    path = os.path.join(det_path,name+".png")
    print("saving mask: %s.png"%name)
    cv.imwrite(path,mask)
    return


''' CV FUNCTIONS '''

def convert_grayscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def bilateral_filter(img, d_val=9,color=100,space=100,border=cv.BORDER_REPLICATE):
    filtered_img = cv.bilateralFilter(img, d=d_val,sigmaColor=color,sigmaSpace=space,borderType=border)
    return filtered_img

def apply_clahe(img, limit=1, grid=(8,8)):
    if len(img.shape) != 2:
        img = convert_grayscale(img)
        print("Converted to Grayscale")
    
    clahe = cv.createCLAHE(clipLimit=limit, tileGridSize=grid)
    improved_hist_img = clahe.apply(img)

    return improved_hist_img

def erode_img(img, window_size = 5, iter=2):
    if window_size == 5:
        kernel = RECT_KERNEL_5X5
    else:
        kernel = RECT_KERNEL_3X3

    eroded_img = cv.erode(img, kernel, iterations = iter)

    return eroded_img

def dilate_img(img, window_size = 5, iter = 2):
    if window_size == 5:
        kernel = RECT_KERNEL_5X5
    else:
        kernel = RECT_KERNEL_3X3
    
    dilated_img = cv.dilate(img, kernel, iterations = iter)

    return dilated_img

def img_thresholding(img, th=60, color=255):

    _, th_img = cv.threshold(img,th,color,cv.THRESH_BINARY_INV)

    return th_img

def morphology_ex(img, type, window_size = 3, iter = 2):
    if window_size == 5:
        kernel = RECT_KERNEL_5X5
    else:
        kernel = RECT_KERNEL_3X3
    
    if type == "close":
        morph = cv.MORPH_CLOSE
    elif type == "open":
        morph = cv.MORPH_OPEN

    morphed_img = cv.morphologyEx(img, morph, kernel, iterations = iter) 

    return morphed_img

def canny_detection(img,th1=45, th2=45,aperture=3,l2_grad=False):

    canny_edges = cv.Canny(img,threshold1=th1, threshold2=th2,apertureSize=aperture,L2gradient=l2_grad)

    return canny_edges

def find_contours(img):

    hierarchy, contours = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    return hierarchy, contours

def get_gallbladder_ctr(ctr, top_n = 5, min_area = 10000, max_area = 150000):

    ctr = sorted(ctr, key=cv.contourArea) 
    big_ctr = ctr[-top_n:]
    filtered_big_ctr = []
    for c in big_ctr:
        area = cv.contourArea(c)
        if area > min_area and area < max_area:
            filtered_big_ctr.append(c)
    
    filtered_big_ctr = sorted(filtered_big_ctr, key=cv.contourArea)
    gb_ctr = filtered_big_ctr[-1]

    return gb_ctr

def approximate_contour(gb_ctr, sample_size = 33):

    poly_mask_gb = []    
    for i in range(int(len(gb_ctr)/sample_size)):
        poly_mask_gb.append(gb_ctr[i*sample_size])
    poly_mask_gb.append(gb_ctr[-1])
    poly_mask_gb = np.array(poly_mask_gb)

    return poly_mask_gb

def draw_contour(mask_ctr, height, width):

    canvas = np.zeros((height, width))
    cv.drawContours(canvas, [mask_ctr], -1, (255,255,255), -1)

    return canvas

def generate_mask(img):

    print("\n\n$$$----Reading Image", img)
    orig_image = cv.imread(img)
    (H,W,C) = orig_image.shape

    smooth_img = bilateral_filter(orig_image)
    show_image(smooth_img, "NOISE_REMOVED")

    grayscale_img = convert_grayscale(smooth_img)
    inc_contrast_img = apply_clahe(grayscale_img)
    inc_contrast_img = bilateral_filter(inc_contrast_img)
    show_image(inc_contrast_img, "CONTRAST_IMPROVED")

    erode_im = erode_img(inc_contrast_img, iter=2)
    show_image(erode_im)

    th_img = img_thresholding(erode_im)
    #show_image(th_img, "THRESHOLD")

    close_th = morphology_ex(th_img, type="close", iter=2)
    erode_close_th = erode_img(close_th, iter=2)
    #show_image(erode_close_th, "CLOSE_ERODE_THRESHOLD")

    contours, _ = find_contours(erode_close_th)
    gb_ctr = get_gallbladder_ctr(contours)
    enhanced_gb_ctr = approximate_contour(gb_ctr, sample_size = 32)
    
    gb_mask = draw_contour(enhanced_gb_ctr, H, W)
    gb_mask = dilate_img(gb_mask, iter=3)
    #show_image(gb_mask, "GALLBLADDER_MASK")
    
    save_mask(gb_mask, img.split("/")[-1][:-4])

    return

if __name__ == "__main__":
     
    parser = argparse.ArgumentParser(description='Grayscale Mask Generation Script for segmentation of Gall Bladder Images')
    parser.add_argument('-i', '--img_path', type=str, default='img', required=True, help="Path for the image folder")
    parser.add_argument('-d', '--det_path', type=str, default='det', required=True, help="Path for the detected masks folder")
    
    args = parser.parse_args()
    
    img_path = args.img_path
    det_path = args.det_path
    print("image folder:", img_path)
    print("det folder:", det_path)
    images = glob(os.path.join(img_path,"*"))
    print("Total # images to process =", len(images))

    #for img in images:
    for img in images:
        generate_mask(img)