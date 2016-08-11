"""
SEE RoofEst_EDGE.py as the latest script
----
The following script is deprecated:
Author: Dingchao Zhang
Created: Aug 9, 2016
Script to estimate roof area using 2d data, manual Height masking, median filetering, etc.
This script is for running on LOCAL with opencv3
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
import csv
import time
#%matplotlib inline

# MODE EDGE is on edgenode which has opencv2, MODE LOCAL is on local mac which has opencv3
if cv2.__version__[0] == '2':
	MODE = 'EDGE'
	
elif cv2.__version__[0] == '3':
	MODE = 'LOCAL'

# SET UP OS
if MODE == 'EDGE':
	os.chdir(r'/san-data/usecase/skyscout/ARI-HackWeek')
elif MODE == 'LOCAL':
	os.chdir(r'/Users/ejlq/Documents/ARI-HackWeek')

def mask_thresh(img):
    
    """
    Apply mask using BGR and Height

    """
    lower_red = np.array([15,114,209]) # manually set lower BGR limit -- TODO:find a way to auto set
    upper_red = np.array([200,200,253]) # manually set upper BGR limit
    mask = cv2.inRange(img,lower_red,upper_red) # create a mask using the color limits
    masked = cv2.bitwise_and(img,img,mask=mask) # apply the mask to image, this will leave the pixels not falling
                                                # into the limits all to black
    return masked
    
    
def read_projdsm(filename):
    """
    Read dsm/project_dsm.tif and return project_dsm read as an array
    containing height info
    
    """
    
    project_pixel = gdal.Open(filename,0)
    project_dsm_1 = project_pixel.GetRasterBand(1)
    project_dsm = project_dsm_1.ReadAsArray()
    
    return project_dsm

def mask_height(img,project_dsm,thresh = 6):
    """
    Apply a height mask, black out certain pixels of img based on mask
    Note: the mask has to be the same dim with img, otherwise will be error out
    """
    heightMask = project_dsm < thresh
    img[heightMask] = [0,0,0]
    
    return img,heightMask

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
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY) # Binary Thresholding 
#     thresh2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)  Gassian Adaptive thresholding not working
    
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
    if MODE == 'LOCAL':
    	print 'yes'
    
	if MODE == 'LOCAL' :
		im2, contours_1, hierarchy = cv2.findContours(median, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		return contours_1
	elif MODE == 'EDGE':
		contours, _ = cv2.findContours(median, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    	return contours

def vis_imgprocess(img,masked,thresh1,median,contours,index):

    """

    Gerenrates visualizations of the intermediate and final img processes

    """    

    img_dummy = median.copy()

    plt.subplot(231), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(232), plt.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)), plt.title('Masked')
    plt.subplot(234), plt.imshow(thresh1), plt.title('Thresholded')
    plt.subplot(235), plt.imshow(median),plt.title('Median Filtered')
    
    # Tried the following to create individual contour, didn't work
    #cv2.drawContours(img_dummy, contours, index, (0, 255, 0), 3)
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

def max_contour(contours):
    """
    find the largest contour, return its number of pixels and index
    
    """
    count,index = max([(cv2.contourArea(v),i) for i,v in enumerate(contours)])
    
    print 'The number of pixels in the largest contour is:', count
    print 'The index of contour which is the largst is:' , index
    
    return count,index
    
def compute_area_2d(contours,index,pixW = 0.00406,pixH = 0.00406,s2rRatio = 1.054):

    """

    Compute area using 2d info

    """ 
    ##DICTIONARY COMPREHENSION EFFICIENT
    
    #c,index = max([(cv2.contourArea(v),i) for i,v in enumerate(contours)])
    
    
    sqM2sqF = 10.7639
    area = cv2.contourArea(cv2.convexHull(contours[index]))*pixW*pixH*s2rRatio*sqM2sqF
    return area

def create_groundTruth(targetCSV):
    """
    Read ARI-target file, return a dictionary with key as house name, actual
    area as item
    
    """
    with open(targetCSV, mode='r') as infile:
        reader = csv.reader(infile)
       #  with open('coors_new.csv', mode='w') as outfile:
#             writer = csv.writer(outfile)
        mydict = {rows[0]:rows[1] for rows in reader}
        
    return mydict
    
def evaluate(actual_dict,house,prediction):
    """
    Compute RMSE, perc_error
    """
    
    house = house +'.pdf'
    try:
        target = float(actual_dict[house])
    except:
        target = 999.12345
    RMSE = np.sqrt(np.mean((prediction-target)**2))
    perc_error = abs(prediction - target)/target
    print 'actual roof sqft is:', target
    print 'RMSE is:', RMSE
    print 'perc_error:', perc_error
    
    return RMSE,perc_error


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






# def pipeline_rgb(filename):
# 	"""
# 	 Pileline all processing functions together using rgb mask only
# 	"""
# 	img = cv2.imread(filename) # read img
# 	masked = mask_thresh(img) # mask img using RGB    
# 	thresh1 = gray_thresh(masked) # grayscale and threshold masked img
# 	median = filtering(thresh1) # median filtering 
# 	contours = create_contours(median) # create contours
# 	c,index = max_contour(contours) # get the largest contour pixels and index
# 	vis_imgprocess(img,masked,thresh1,median,contours,index) # visualize

# 	pixW,pixH = georef(filename) # get pixel width and height in meters

# 	areaEst = compute_area_2d(contours,index,pixW = pixW,pixH = pixH,s2rRatio = 1.054)

# 	print "estimated area square feet is using RGB mask only is \n", areaEst
	
# 	return 0
	
    
def pipeline_height(filename,dsm):
	"""
	 Pileline all processing functions together using height mask only
     filename: color_relief.tif
     dsm: project_dsm.tif
     now the order is apply height mask then rgb mask
     good to try a different order --TODO!!
     
	"""
	start_time = time.time()

	img = cv2.imread(filename) # read img
	project_dsm = read_projdsm(dsm) # read project_dsm.tif file
	masked,heightMask = mask_height(img,project_dsm) # mask img using height mask only  
	#thresh1 = gray_thresh(masked) # grayscale and threshold masked img
    #thresh1 = gray_thresh(masked) # grayscale and threshold masked img
	thresh1 =  cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)  
	median = filtering(thresh1) # median filtering 
	contours = create_contours(median) # create contours
	c,index = max_contour(contours) # get the largest contour pixels and index
	#vis_imgprocess(img,masked,thresh1,median,contours,index) # visualize

	pixW,pixH = georef(filename) # get pixel width and height in meters

	areaEst = compute_area_2d(contours,index,pixW = pixW,pixH = pixH,s2rRatio = 1.054)
	exe_time = time.time() - start_time
	print "estimated area square feet is using height mask only is \n", areaEst
	print "seconds ",exe_time
	return areaEst,exe_time
	
    
    
# def pipeline_combined(filename,dsm):
# 	"""
# 	 Pileline all processing functions together using both height and rgb mask
# 	"""
# 	img = cv2.imread(filename) # read img
# 	img2 = mask_thresh(img) # mask img using RGB 
# 	project_dsm = read_projdsm(dsm) # read project_dsm.tif file
# 	masked,heightMask = mask_height(img2,project_dsm) # mask img using height mask only  
# 	thresh1 = gray_thresh(masked) # grayscale and threshold masked img
# 	median = filtering(thresh1) # median filtering 
# 	contours = create_contours(median) # create contours
# 	c,index = max_contour(contours) # get the largest contour pixels and index
# 	vis_imgprocess(img,masked,thresh1,median,contours,index) # visualize

# 	pixW,pixH = georef(filename) # get pixel width and height in meters

# 	areaEst = compute_area_2d(contours,index,pixW = pixW,pixH = pixH,s2rRatio = 1.054)

# 	print "estimated area square feet is using combined mask is \n", areaEst
	
# 	return 0
	
    
def main():  
    start_time = time.time()
    
# 	if MODE = 'LOCAL' :
# 		print 'yes'
    	
    if MODE == 'LOCAL':
    	rootdir = '/Users/ejlq/Documents/ARI-HackWeek/training'
    	target_dict = create_groundTruth('ARI-targets.csv')
    elif MODE == 'EDGE':
    	rootdir = '/san-data/usecase/skyscout/ARI-HackWeek/training'
    	target_dict = create_groundTruth('/home/ejlq/ARI-targets.csv')
    results = []
    
    
#     with open('ARI-targets.csv', mode='r') as infile:
#         reader = csv.reader(infile)
#         with open('coors_new.csv', mode='w') as outfile:
#             writer = csv.writer(outfile)
#             target_dict = {rows[0]:rows[1] for rows in reader}


    
    for subdir, dirs, files in os.walk(rootdir):

        colorfile = []
        projectfile = []
        
        for infile in files: 
            
            result = []
            
            if infile.endswith('color_relief.tif'):
                
                filename = subdir+'/'+infile
                colorfile.append(filename)
               
                
            if infile.endswith('project_dsm.tif'):
                filename = subdir+'/'+infile
                projectfile.append(filename)
                
        if (len(colorfile) == 1) & (len(projectfile) == 1):
            f_name = colorfile[0].split('/')[-3]
            print '----computing the following house now:' , f_name
            est, exe_time= pipeline_height(colorfile[0],projectfile[0])
            RMSE,perc_error = evaluate(target_dict,f_name,est)
            
            result.append(f_name)
            result.extend([est,exe_time,RMSE,perc_error])
            results.append(result)
            
# 	with open("output.csv", "wb") as f:
#     		writer = csv.writer(f)
#     		writer.writerows(results)
    
    if MODE == 'LOCAL' :
    	with open("output.csv", "wb") as f:
    		writer = csv.writer(f)
    		writer.writerows(results)
    elif MODE == '/home/ejlq/EDGE' :
    	with open("output.csv", "wb") as f:
    		writer = csv.writer(f)
    		writer.writerows(results)
        
    print 'results saved to output.csv'
    print 'in total takes the followig minutes to run', (time.time() - start_time)/60
    return results
        
                

if __name__ == '__main__':
    main()