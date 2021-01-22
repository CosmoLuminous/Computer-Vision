'''
**********************************************
# TITLE: ASSIGNMENT 2: IMAGE MOSAIC CREATION # 
# #Panorama #Stitching #Blending #Homography #
# #GraphCut #PyramidBlending #SIFT #Warp     #
# COURSE: COL780 - COMPUTER VISION           #
# INSTRUCTOR: PROF. CHETAN ARORA             #
# AUTHOR: AMAN BHARDWAJ                      #
# DATE: 20 NOV 2020                          #
**********************************************
'''
''' IMPORT PACKAGES '''
import os
import cv2 as cv
from glob import glob
import numpy as np
import argparse
#import matplotlib.pyplot as plt
#from time import time
import maxflow as mf

""" HELPER FUNCTIONS """
def save_mosaic(fileName, mos):
    '''Save Mosaic in the same folder as createmosaic.py file.'''
    print("Saving --->", fileName)
    cv.imwrite(fileName, mos)
    return

# def show_image(img, title='Default', save=False):
#     '''Display and save image'''
#     if img is None:
#         print('Could not find image to show.')
#     else:        
#         print("\n\n%%%% IMAGE: {}, SHAPE: {} %%%%".format(title, img.shape))
#         fig = plt.figure(num=0, figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
#         plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#         plt.axis('off')
#         plt.show()
#         if save:
#             fig.savefig("plots/"+title+".png", dpi=300, format="png")       
#     return
def show_image(img, title='Default', save=False):
    #dummy function for replacement of original show_image function
    #as we are not allowed to use matplotlib.
    print("SHOW_IMAGE Called")
    return

def read_image(fileName):
    '''Read image from the specified path.'''
    print("Reading Image --->", fileName)
    img = cv.imread(fileName)
    return img

def hconcat_images(image_list):
    '''Concat images horizontally'''
    return cv.hconcat(image_list)

class ImgProcessing:
    '''
    CLASS: ImgProcessing
    Contains functions to prepare the images for mosaicing.
    Operations:
    1. Image Resize
    2. Brightness Equaliation
    3. Clahe Contrast Stretching
    '''
    def image_resize(self, image, value=1000,inter = cv.INTER_AREA):
        '''Resize image to a maximum dim = value'''
        dim = None
        (h, w) = image.shape[:2]
        height = width = None
        if h > w:
            height = value
        else:
            width = value

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)        
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv.resize(image, dim, interpolation = inter)
        return resized

    def brightness_equalize(self, img):
        '''
        Brightness correction by histogram equalization.
        '''
        img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])
        img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
        return img_output

    def apply_clahe(self, img, limit=1.2, grid=(8,8)):
        '''
        CLAHE:
        Histogram Equalization and Contrast enhancement
        '''    
        img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

        clahe = cv.createCLAHE(clipLimit=limit, tileGridSize=grid)
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])    
        img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)        
        img_output = cv.normalize(img_output, None, 0, 255, cv.NORM_MINMAX)
        
        return img_output


class RegisterImage:
    '''
    CLASS: RegisterImage
    It registers image features and key point for mosaicing
    
    '''
    
    def cvt_gray(self, img):
        '''Convert BRG Image to Grayscale'''
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    def orbs_magic(self, img, show=False):
        '''
        SIFT: Descriptor
        Detect, extract features for img1 and img2.
        Using SIFT as the detector/extractor
        previously tried orb descriptor as well but it was identifying way less points than sift.
        Therefore the name of function is orbs_magic ;)
        '''
        #init SIFT
        orb = cv.SIFT_create(nfeatures = 500)
        #detect keypoints
        img = self.cvt_gray(img)
        keyPoints = orb.detect(img, None)
        #get descriptor
        keyPoints, imgDescriptors = orb.compute(img, keyPoints)

        if show:
            #draw detected keypoints on image and show
            img2 = img
            img2 = cv.drawKeypoints(img,keyPoints,img2,color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            show_image(img2, "SIFT_DETECTOR_"+str(z), True)
        return keyPoints, imgDescriptors

    def flann_matcher(self, d1, d2, K=2):
        '''
        Performs feature matching between key points of img1 and img2
        Adapted from: https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
        '''
        search_params = {'checks': 50} #default params
        index_params = {'algorithm': 1, 'table_number': 6, 'key_size': 12, 'multi_probe_level': 1, 'trees': 5} #algo = FLANN_INDEX_LSH
        #init flann
        flann = cv.FlannBasedMatcher(index_params, search_params)
        knn_matches = flann.knnMatch(d1, d2, k=K)

        return knn_matches

    def get_good_matches(self, matches, thresh=0.8):
        '''
        Of all the matches identify the good matches based on threshold value = thresh
        '''
        good_matches = []
        for m in matches:
            if m[0].distance < thresh * m[1].distance:
                good_matches.append(m[0])
        return good_matches   

    def draw_matches(self, img1, img2, k1, k2,  g_matches, draw=False):
        '''
        helper function to draw the matches between the two images.
        '''
        img1 = self.cvt_gray(img1)
        img2 = self.cvt_gray(img2)
        match_canvas = cv.drawMatches(img1, k1, img2, k2, g_matches, None, 
                                      flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        if draw:
            show_image(match_canvas, "KNN_MATCHES_"+str(z), True)
        return

    def est_homography(self, k1, k2, confident_matches, min_matches=20):
        '''
        Estimate Homography Matrix (3x3) between both images
        RANdom SAmple Consensus(RANSAC) algorithm proposed by Fischler andBolles [1] is 
        a general parameter estimation approach designed to cope with a large proportion of outliers in the input data. 
        '''
        print("# Good Matches ={}, # Minimum Points ={}".format(len(confident_matches), min_matches))
        if len(confident_matches) >= min_matches:
            points_img1 = [k1[m.queryIdx].pt for m in confident_matches]
            points_img2 = [k2[m.trainIdx].pt for m in confident_matches]
            points_img1 = np.array(points_img1, dtype=np.float32).reshape(-1, 1, 2)
            points_img2 = np.array(points_img2, dtype=np.float32).reshape(-1, 1, 2)
            H, mask = cv.findHomography(points_img2, points_img1, cv.RANSAC, 5.0)
            inliers = list(mask.ravel())
            flag = "good_matches"
        else:
            print("Number of Good Matches are less than Minimum Points set for Homography estimation.")
            H = inliers = None
            flag = "not_enough_matches"
        return H, inliers, flag

class ImageBlending:
    '''
    CLASS: ImageBlending
    Once the Image Features have been registered and Homography has been calculated. Images are ready for blending.
    It performs following tasks:
    1. Image Warp 
    2. Energy flow Calculation
    3. Graph Cut
    4. Pyramid Blending
    '''
    def warp_image(self, i1, i2, H, show=False):
        '''
        Keeping one image as reference it warps the other image based on the H Matrix calculated.
        '''
        h1, w1 = i1.shape[:2]
        h2, w2 = i2.shape[:2]    
        
        gray = cv.cvtColor(i1, cv.COLOR_BGR2GRAY)
        thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
        i1_mask = thresh1
        gray = cv.cvtColor(i2, cv.COLOR_BGR2GRAY)
        thresh2 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
        
        mosaic = cv.warpPerspective(i2, H, (w1,h1), dst=i1.copy(), borderMode=cv.BORDER_TRANSPARENT)
        i2_mask = cv.warpPerspective(thresh2, H, (w1,h1), borderMode=cv.BORDER_TRANSPARENT)
        i2_mask = cv.threshold(i2_mask, 254, 255, cv.THRESH_BINARY)[1]
        i2_warp = cv.warpPerspective(i2, H, (w1,h1))
        
        if show:
            show_image(mosaic, "MOSAIC_"+str(z), True)
            show_image(hconcat_images([i1_mask,i2_mask]), "M1_&_M2 "+str(z), True) 
            show_image(i2_warp, "I2_Warp "+str(z), True)
        return mosaic, i1_mask, i2_mask, i2_warp
    
    def get_images_intersect(self, m1, m2, mos):
        '''
        Helper function to find the intersection mask of m1 and m2
        '''
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
        #m1 = cv.erode(m1, kernel, iterations = 1)
        #m2 = cv.erode(m2, kernel, iterations = 1)
        
        common_mask = cv.bitwise_and(m1, m2)
        common_mos = cv.bitwise_and(mos, mos, mask=common_mask)
        
        return common_mask, common_mos
    
    def get_energy_map(self, cmos):
        '''
        Calculates energy flow based on x and y deravatives of images.
        '''
        grad_u = cv.Sobel(cmos,cv.CV_32F,0,1,ksize=3)
        grad_v = cv.Sobel(cmos,cv.CV_32F,1,0,ksize=3)
        print(grad_u.shape , grad_v.shape)
        convolved =  grad_u + grad_v

        # We sum the energies in the red, green, and blue channels
        energy_map = convolved.sum(axis=2)
        energy_map[energy_map > 0.7] = 1.0
        energy_map[energy_map <0.2] = 0.0
        
        #show_image(energy_map, "ENERGY MAP2")
        return energy_map
    
    def get_extreme_columns(self, cmask):
        '''
        Helper function to get to bounding box for common region to be used for graph cut.
        '''
        h, w = cmask.shape
        top = 0
        bottom = h
        col_leftmost = 0
        col_left = 0
        col_right = w
        for c in range(w):
            v = cmask[:,-c]
            if np.max(v) > 0:
                col_right = w-c-1
                break
                
        for i in range(col_right,0,-1):
            v = cmask[:,i]
            if int(np.sum(v)) == 0:
                col_leftmost = i+1
                break
        for j in range(h):
            row = cmask[-j,:]
            if np.max(row) > 0:
                bottom = h-j-1
                break
                
        for i in range(col_leftmost, col_right):
            v = cmask[bottom, i]
            if v > 0:
                col_left = i+1
                break
        
        for i in range(h):
            row = cmask[i,:]
            if np.max(row) > 0:
                top = i
                break
        return int((col_left+col_leftmost)/2-(col_left-col_leftmost)/3.5), col_right, top, bottom
    
    def get_top_right(self, mos):
        '''helper function'''
        h, w = mos.shape[:2]
        gray = cv.cvtColor(mos, cv.COLOR_BGR2GRAY)
        mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
        right = w
        top = 0
        for c in range(w):
            v = mask[:,-c]
            if np.max(v) > 0:
                right = w-c-1
                break        
        for i in range(h):
            row = mask[i,:]
            if np.max(row) > 0:
                top = i
                break        
        return right, top
    
    def graph_cut(self, cmask, cmos, left_img, right_img, i2_mask, mos):
        '''
        Graph Cut Implementation:
        Estimate seam between img1 and img2 by Max flow - Min Cut Algorithm
        Used for PyMaxflow library for computing Min Cut.
        It's a python wrapper around C++ based implementation of Kolmogorov algorithm
        '''
        #find overlap region
        left_col, right_col, top_row, bottom_row = self.get_extreme_columns(cmask)

        c_left = cv.bitwise_and(left_img,left_img, mask= cmask.astype(np.uint8))
        c_right = cv.bitwise_and(right_img,right_img, mask= cmask.astype(np.uint8))
                
        #calculate energy map    
        c_int_diff = np.abs(cv.cvtColor(c_left, cv.COLOR_BGR2GRAY).astype(np.float32) - 
                          cv.cvtColor(c_right, cv.COLOR_BGR2GRAY).astype(np.float32))/16.0
        cmos_emap = c_int_diff[:, left_col: right_col]
        #show_image(c_int_diff.astype(np.float32)/3.5, "ENERGY_MAP_"+str(z), True)
        
        #graph init
        dim = cmos_emap.shape[:2]
        graph = mf.Graph[float]()
        inf = np.inf
        nodes = graph.add_grid_nodes(dim)        
        
        #connect edges
        # Source node connected to leftmost non-terminal nodes.
        leftmost_nodes = nodes[:, 0]
        graph.add_grid_tedges(leftmost_nodes, inf, 0)
        # Sink node connected to rightmost non-terminal nodes.
        rightmost_nodes = nodes[:, -1]
        graph.add_grid_tedges(rightmost_nodes, 0, inf)
        
        # Edges pointing right
        structure = np.zeros((3,3))
        structure[1,2] = 1
        weights_fwd = np.abs(np.roll(cmos_emap, -1, axis=1) + cmos_emap)
        graph.add_grid_edges(nodes, structure=structure, weights=weights_fwd, symmetric=True)

        # Edges pointing down
        structure = np.zeros((3,3))
        structure[2,1] = 1
        weights_down = np.abs(np.roll(cmos_emap, -1, axis=0) + cmos_emap)
        graph.add_grid_edges(nodes, structure=structure, weights=weights_down, symmetric=False)  
        
        #find max flow equivalent to min cut
        flow = graph.maxflow()
        print('maxflow: {}'.format(flow))
        segments = graph.get_grid_segments(nodes)
        min_cut = np.logical_not(segments).astype(np.uint8)
        #fig = plt.figure(1)
        #plt.imshow(min_cut, cmap="gray")
        #plt.axis('off')
        #plt.show()
        #fig.savefig("./plots/min_cut_{}.png".format(str(z)), dpi=300, format="png") 
        
        #apply min cut to left and right overlaps
        o_left = left_img[:, left_col: right_col]
        o_right = right_img[:, left_col: right_col]
        o_left_dash = cv.bitwise_and(o_left,o_left, mask= min_cut.astype(np.uint8))
        o_right_dash = cv.bitwise_and(o_right,o_right, mask= np.logical_not(min_cut).astype(np.uint8))        
        #show_image(hconcat_images([o_left_dash, o_right_dash]), "LEFT_AND_RIGHT_OVERLAP_".format(str(z), True))

        #create new mosaic 
        o_new_mosaic = o_left_dash + o_right_dash
        new_mosaic = mos
        new_mosaic[:, left_col: right_col] = o_new_mosaic
        new_mosaic[:,0: left_col] = left_img[:,0: left_col]
        new_mosaic[:,right_col: ] = right_img[:,right_col: ]
#         new_mosaic[-200:,:] = right_img[-200:,:]
        new_mosaic[0:200,:] = right_img[0:200,:]
        ri, to = self.get_top_right(new_mosaic)
        new_mosaic = new_mosaic[to:-200,200:ri]
        #show_image(new_mosaic, "NEW_MOSAIC_"+str(z), True)
        
#         new_left[:, left_col: right_col] = o_left_dash
#         new_left[:,0: left_col] = left_img[:,0: left_col]
        
#         new_right_dash = new_right.copy()
#         new_right_dash[:, left_col: right_col] = o_right_dash
#         new_right_dash[:,right_col: ] = new_right[:,right_col: ]
#         new_right_dash[-200:,:] = new_right[-200:,:]
#         new_right_dash[0:200,:] = new_right[0:200,:]
#         new_mosaic_dash = new_right_dash + new_left
#         show_image(hconcat_images([new_left, new_right_dash]), "NEW_LEFT_RIGHT")
#         show_image(new_mosaic, "NEW_MOSAIC_"+str(z), True)

        return new_mosaic
    

    def gaussian_pyramid(self, img, l):
        '''
        Helper function to calculate Gaussian Pyramid
        '''
        GP = list()
        GI = img.copy()
        for i in range(l):
            if i==0:                
                GP.append(GI)
            GI =  cv.pyrDown(GI)
            GP.append(GI)
        return GP
    
    def laplacian_pyramid(self, GP, l):
        '''
        helper function to calculate laplacian pyramid
        '''
        LP = list()
        for i in range(l-1, 0, -1):
            if i == l-1:    
                LP.append(GP[i])
            G = GP[i-1]
            gh, gw = G.shape[:2]
            L = cv.pyrUp(GP[i], dstsize=(gw,gh))
            lp = cv.subtract(G,L)
            LP.append(lp)
        return LP
    
    def blend(self, lp1, lp2, m,l):
        '''
        helper function to blend gaussian and laplacian pyramids difference
        '''
        LS = []
        for i, lp in enumerate(zip(lp1, lp2, m)):
            l1, l2, mask = lp
            #print(l1.shape, l2.shape, mask.shape)
            l1 = cv.bitwise_and(l1,l1, mask=np.logical_not(mask).astype(np.uint8))
            l2 = cv.bitwise_and(l2,l2, mask=mask)
            ls =  l1+l2 
            LS.append(ls)
        img_blend = LS[0]
        for i in range(1, l, 1):
            L = LS[i]
            lh, lw = L.shape[:2]
            
            img_blend = cv.pyrUp(img_blend, dstsize=(lw,lh))
            img_blend = cv.add(img_blend,L)
            
        return img_blend
        
    def pyramid_blending(self, mos, pyr_levels=6):
        '''
        Pyramid Blending Implementation:
        ToDo: improve pyramid blending coz graph cut seam is still visible
        '''
        mask = np.zeros(mos.shape[:2], dtype=np.uint8)
        mask1 = np.zeros(mos.shape[:2], dtype=np.uint8)
        w = int(mos.shape[1]/2)
        mask[:,w:] = 1
        
        #im2 = cv.bitwise_and(mos, mos, mask=mask)
        #im1 = cv.bitwise_and(mos, mos, mask=mask1)
        im1 = mos
        im2 = mos
        
        GP1 = self.gaussian_pyramid(im1, pyr_levels)
        GP2 = self.gaussian_pyramid(im2, pyr_levels)
        GM = self.gaussian_pyramid(mask, pyr_levels)
        
        GM_rev = GM.copy()
        GM_rev.reverse()
        GM_rev.pop(0)
        
        LP1 = self.laplacian_pyramid(GP1, pyr_levels)
        LP2 = self.laplacian_pyramid(GP2, pyr_levels)
        
        blended_image = self.blend(LP1, LP2, GM_rev, pyr_levels)
        #show_image(blended_image, "PYRAMID_BLEND_"+str(z), True)
        return blended_image
    
def create_mosaic(imgs, contrast_imp=True):
    #read images
    Im1 = imgs[0]
    Img1_ = read_image(Im1)
    Im2 = imgs[1]    
    Img2_ = read_image(Im2)   
    #show_image(hconcat_images([Img1_, Img2_]), "INPUT IMAGES")
    
    #image preprocessing
    ip = ImgProcessing() #instance of Image Processing Class
    Img1_ = ip.image_resize(Img1_)    
    Img2_ = ip.image_resize(Img2_)
    if contrast_imp:
        Img1 = ip.apply_clahe(Img1_)
        Img2 = ip.apply_clahe(Img2_)
    else:
        Img1 = Img1_
        Img2 = Img2_
    print(Img1.shape, Img2.shape, type(Img1), type(Img2))
    #show_image(Img1, "CLAHE BRIGHTNESS IMPROVED IMAGES")
    #show_image(Img2)
    Img1 = cv.copyMakeBorder(Img1,200,200,200,600, cv.BORDER_CONSTANT) 
    
    #get keypoints and descriptors
    reg = RegisterImage()
    kp1, des1 = reg.orbs_magic(Img1, False)
    kp2, des2 = reg.orbs_magic(Img2, False)
    
    #feature matching by FLANN based Matcher
    knn_matches = reg.flann_matcher(des1, des2)
    confident_matches = reg.get_good_matches(knn_matches, 0.8)
    reg.draw_matches(Img1, Img2, kp1, kp2, confident_matches, False)
    
    #estimate homography using RANSAC
    H, inliers, flag = reg.est_homography(kp1, kp2, confident_matches, 40)
    
    if flag == "good_matches":
    #Graph cut and pyramid blending
        blend = ImageBlending()
        mosaic_pre, M1, M2, Img2_warp = blend.warp_image(Img1, Img2, H, False)
        c_mask, c_mos = blend.get_images_intersect(M1, M2, mosaic_pre)

        graphcut_mos = blend.graph_cut(c_mask, c_mos, Img1, Img2_warp, M2,mosaic_pre)
        graphcut_mos = ip.image_resize(graphcut_mos, value=2000)
        #graphcut_mos = ip.apply_clahe(graphcut_mos, limit=1)
        pyramid_blend = blend.pyramid_blending(graphcut_mos, pyr_levels=3)
        #pyramid_blend = ip.apply_clahe(pyramid_blend, limit=0.2) 
        #pyramid_blend = ip.image_resize(pyramid_blend, value=2000)
        pyramid_blend = cv.normalize(pyramid_blend, None, 0, 255, cv.NORM_MINMAX)
        return graphcut_mos, pyramid_blend
    else:
        return Img1, Img1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grayscale Mask Generation Script for segmentation of Gall Bladder Images')
    parser.add_argument('-i', '--input_path', type=str, default='./val_set/1', required=True, help="Image Folder Path")
    args = parser.parse_args()
    input_path = args.input_path
    print(input_path)
    images = glob(os.path.join(input_path, "*"))
    print("Folder Content:",images)
    z = ""
    if len(images) == 2:
        print("\nPerfect Input! Let me stitch them for you! ;)")
        graphcut_mosaic, pyramidblend_mosaic = create_mosaic(images)
        image_name = input_path.split("/")[-1]
        z = image_name
        save_mosaic(image_name+"_graph_cut_mosaic.png",graphcut_mosaic)
        save_mosaic(image_name+"_pyramid_blend_mosaic.png",pyramidblend_mosaic)
    elif len(images) == 1:
        print("\nOnly 1 image! Fine. mosaic == input. I can do better! Put 2 images and let me show you what i can do! :P")
        save_mosaic(images[0].split('/')[-2], cv.imread(images[0]))
    elif len(images) == 0:
        print("\nCome on!! You want to create mosaic with no images in folder??? :P")
    elif len(images) > 2:
        print("Wooo! Hold on buddy! You are getting too ambitious. Folder contains more than 2 images. :D")
    else:
        print("\nSome problem with input folder :( Please check.")



