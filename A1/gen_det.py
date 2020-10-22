import os
import cv2 as cv
import json
from glob import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt

def show_image(img, name="default",save=False):
    if img is None:
        print('Could not open or find the image: ', args.input)
        exit(0)
    else:
        '''cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
        print("shape", img.shape)
        fig = plt.figure(0)
        plt.imshow(img, cmap='gray')
        plt.show()
        if save:
            fig.savefig("plots/"+name+".png", dpi=300, format="png")        
    return
def save_mask(name, mask):
    pth = os.path.join(det_path,name+".png")
    print("saving mask:",pth)
    cv.imwrite(pth,mask)


def generate_mask(img):
    print("\n\n$$$Reading Image", img)
    orig_image = cv.imread(img)
    (H,W,C) = orig_image.shape
    print("Original Image", orig_image.shape)
    #show_image(orig_image)
    
    image = cv.bilateralFilter(orig_image, d=9,sigmaColor=100,sigmaSpace=100,borderType=cv.BORDER_REPLICATE)
    print("\n\nImage Smoothing")
    #show_image(image)
    clahe = cv.createCLAHE(clipLimit=1, tileGridSize=(8,8))
    imp_cont_image = clahe.apply(cv.cvtColor(image,cv.COLOR_BGR2GRAY))
    print("Improved contrast")
    
    imp_cont_smt_image = cv.bilateralFilter(imp_cont_image, d=9,sigmaColor=100,sigmaSpace=100,borderType=cv.BORDER_REPLICATE)
    #print("Improved contrast + Smoothed")
    
    show1 = cv.hconcat([cv.cvtColor(image, cv.COLOR_BGR2GRAY),imp_cont_image, imp_cont_smt_image])
    #show_image(show1)

    erode_img = cv.erode(imp_cont_smt_image, cv.getStructuringElement(cv.MORPH_RECT,(5,5)), iterations=2) 
    print("Erode Image")
    #show_image(erode_img)

    ret, th_img = cv.threshold(erode_img,60,255,cv.THRESH_BINARY_INV)
    show3 = cv.hconcat([erode_img, th_img])
    #show_image(show3,"thresh",True)
    print("Thresholding")

    #open = cv.morphologyEx(canny_edges, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT,(3,3)), iterations = 1) 
    open_close = cv.morphologyEx(th_img, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(3,3)), iterations = 2) 
    
    open_close = cv.erode(open_close, cv.getStructuringElement(cv.MORPH_RECT,(5,5)), iterations=2)
    show4 = cv.hconcat([th_img, open_close])
    print("Close and Erode operations")
    #show_image(show4, "Th Close", True)

    canny_edges = cv.Canny(open_close,threshold1=45, threshold2=45,apertureSize=3,L2gradient=False)

    show2 = cv.hconcat([open_close, canny_edges])
    #show_image(show2, "Close Canny", True)
    print("Canny Edge Detection")

    contours, hierarchy = cv.findContours(open_close, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    print("Find all contours and Extract Gallbladder")

    #print(len(contours))
    ctr = sorted(contours, key=cv.contourArea) 
    big_ctr = ctr[-5:]
    filtered_big_ctr = []
    for c in big_ctr:
        area = cv.contourArea(c)
        #print("Area",area)
        if area > 10000 and area < 150000:
            filtered_big_ctr.append(c)
    
    filtered_big_ctr = sorted(filtered_big_ctr, key=cv.contourArea)
    gb_ctr = filtered_big_ctr[-1]
    output = np.zeros((H,W))
    poly_mask_gb = []
    sampler = 33
    for i in range(int(len(gb_ctr)/sampler)):
        poly_mask_gb.append(gb_ctr[i*sampler])
    poly_mask_gb.append(gb_ctr[-1])

    cv.drawContours(output, [np.array(poly_mask_gb)], -1, (255,255,255), -1)
    output = cv.dilate(output, cv.getStructuringElement(cv.MORPH_RECT,(5,5)), iterations=3)

    #show_image(output, img.split("/")[-1][:-4], True)
    save_mask(img.split("/")[-1][:-4], output)

    return contours, hierarchy



if __name__ == "__main__":
     
    parser = argparse.ArgumentParser(description='Grayscale Mask Generation Script for segmentation of Gall Bladder Images')
    parser.add_argument('-i', '--img_path', type=str, default='img', required=True, help="Path for the image folder")
    parser.add_argument('-d', '--det_path', type=str, default='det', required=True, help="Path for the detected masks folder")
    
    args = parser.parse_args()
    
    img_path = args.img_path
    det_path = args.det_path
    print("image folder:", img_path)
    print("det folder:", det_path)
    images = glob(os.path.join(img_path,"*.jpg"))
    print("Total # images to process =", len(images))

    #for img in images:
    for img in images:
        ctr, hrcy = generate_mask(img)