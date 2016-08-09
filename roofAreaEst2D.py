"""
Author: Dingchao Zhang
Created: Aug 9, 2016
Script to estimate roof area using 2d data, manual masking, median filetering, etc.
"""


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import gdal                     #part of gdal 1.11
from gdalconst import *         #part of gdal 1.11
import glob
import matplotlib.pyplot as plt
#%matplotlib inline

os.chdir(r'/Users/ejlq/Documents/ARI-HackWeek')

# file_name = 'dsm/color_relief.tif'
# img = cv2.imread(file_name)

def mask_thresh(img):
    
    """
    Apply mask

    """
    lower_red = np.array([15,114,209]) # manually set lower BGR limit -- TODO:find a way to auto set
    upper_red = np.array([200,200,253]) # manually set upper BGR limit
    mask = cv2.inRange(img,lower_red,upper_red) # create a mask using the color limits
    masked = cv2.bitwise_and(img,img,mask=mask) # apply the mask to image, this will leave the pixels not falling
                                                # into the limits all to black
    return masked

def resize(masked):

    """
    Resize(optional), for trial and display fitting into screen mostly
    maybe is more efficient?

    """
 
    # we need to keep in mind aspect ratio so the image does
    # not look skewed or distorted -- therefore, we calculate
    #the ratio of the new image to the old image
    r = 100.0 / masked.shape[1]
    dim = (100, int(masked.shape[0] * r))
    resized = cv2.resize(masked, dim, interpolation = cv2.INTER_AREA) # resize
    
    return resized

def gray_thresh(resized):

    """
    Grayscale and Thresholding

    """
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # Gray scaled
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY) # Thresholding 
    
    return thresh1

def filtering(thresh1,filter_window = 15):
    """
    Apply Median Filter

    """    
    median = cv2.medianBlur(thresh1,filter_window)
    
    return median

def create_contours(median):

    """
    Find contours

    """
    im2, contours, hierarchy = cv2.findContours(median, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def vis_imgprocess(img,masked,thresh1,median,contours):

    """

    Gerenrates visualizations of the intermediate and final img processes

    """    

    img_dummy = median.copy()

    plt.subplot(231), plt.imshow(img), plt.title('Original')
    plt.subplot(232), plt.imshow(masked), plt.title('Masked')
    plt.subplot(234), plt.imshow(thresh1), plt.title('Thresholded')
    plt.subplot(235), plt.imshow(median),plt.title('Median Filtered')
    
    cv2.drawContours(img_dummy, contours, -1, (0, 255, 0), 3)
    plt.subplot(236), plt.imshow(img_dummy), plt.title('Contour')
    plt.show()
    
    
def georef(infile):
    """
    Get pixel width and height in meters

    """    
    geofile = gdal.Open(infile) # open the mosaic image
    if geofile is None:
        print 'Could not open image'
        sys.exit(1)
        
    geoTransf = geofile.GetGeoTransform()
#     xOrigin = geoTransf[0]
#     yOrigin = geoTransf[3]
    pixelWidth = abs(geoTransf[1])
    pixelHeight = abs(geoTransf[5])
    
    return pixelWidth,pixelHeight
    
    
def compute_area_2d(contours,pixW = 0.00406,pixH = 0.00406,s2rRatio = 1.054):

    """

    Compute area using 2d info

    """ 
    ##DICTIONARY COMPREHENSION EFFICIENT
    
    c,index = max([(cv2.contourArea(v),i) for i,v in enumerate(contours)])

    sqM2sqF = 10.7639
    area = cv2.contourArea(cv2.convexHull(contours[index]))*pixW*pixH*s2rRatio*sqM2sqF
    return area

# def handle_args():
#     """
#     command line arguments parser
    
#     """
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', '--directory', action='store',
#                         help='Name of I/O directory')
#     parser.add_argument('-e', '--extension', action='store', default='tif',
#                         help='Extension of files to be read')
#     args = parser.parse_args()
    
#     if args.directory is None:
#         print('Setting I/O directory to current working directory')
#         args.directory = '.'
        
#     return args


def main():
    
#     args = handle_args()
#     filelist = glob.glob(args.directory)
    
    rootdir = '/Users/ejlq/Documents/ARI-HackWeek/training'

    for subdir, dirs, files in os.walk(rootdir):
        for infile in files: 
            if infile.endswith('color_relief.tif'):
                
                print os.path.join(subdir, infile)
                filename = subdir+'/'+infile
                img = cv2.imread(filename) # read img
                masked = mask_thresh(img) # mask img
                thresh1 = gray_thresh(masked) # grayscale and threshold masked img
                median = filtering(thresh1) # median filtering 
                contours = create_contours(median) # create contours

                #vis_imgprocess(img,masked,thresh1,median,contours) # visualize

                pixW,pixH = georef(filename) # get pixel width and height in meters

                areaEst = compute_area_2d(contours,pixW = pixW,pixH = pixH,s2rRatio = 1.054)

                print "estimated area square feet is \n", areaEst

if __name__ == '__main__':
    main()