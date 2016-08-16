"""
Author: Dingchao Zhang
Masking Methods: Jing 
Aaptive Height Threshold :Sripriya
Created: Aug 13, 2016
Script to estimate roof area using 2d data, adaptive Height masking, median filetering, etc.
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
import pandas as pd
from collections import Counter
%matplotlib inline

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

    
def read_projdsm(dsm):
    """
    Read dsm/project_dsm.tif and return project_dsm read as an array
    containing height info
    
    """
    
    geofile = gdal.Open(dsm,0)
    project_dsm_1 = geofile.GetRasterBand(1)
    project_dsm = project_dsm_1.ReadAsArray()
    
    return project_dsm,geofile

def load_orthomosaic(orthom):
    """
    Load orthomosaic file and return rgbArray
    """
    pixelimageDS = gdal.Open(orthom)
    pr_data = pixelimageDS.GetRasterBand(1)
    pg_data = pixelimageDS.GetRasterBand(2)
    pb_data = pixelimageDS.GetRasterBand(3)
    #pa_data = pixelimageDS.GetRasterBand(4)

    r_data = pr_data.ReadAsArray()
    g_data = pg_data.ReadAsArray()
    b_data = pb_data.ReadAsArray()
    #a_data = pa_data.ReadAsArray()

    rgbArray = np.dstack((r_data, g_data, b_data))
    return rgbArray

def hsv_mask(rgbArray):
    """
    Load rgbArray read from orthomosaic file and return a mask for filter green color points
    """
    hsv_rgb = cv2.cvtColor(rgbArray, cv2.COLOR_BGR2HSV)

    ## define range of green color in HSV    
    lower_rgb = np.array([65,20,20])
    upper_rgb = np.array([85,255,255])

    # Threshold the HSV image to create a mask for filter green color points
    mask_rgb= 255 - cv2.inRange(hsv_rgb, lower_rgb, upper_rgb)
    
    return mask_rgb

def adaptive_thresh(cloud_file, offset = 1.5):
    df = pd.read_csv(cloud_file, header=None)
    cloud_data = df.as_matrix()
    zValues = cloud_data[:,2]
    zNum = [np.int(i) for i in zValues]
    freq = Counter(zNum)
    maxZVal = [i[0] for i in freq.most_common() if i[0] > 3][0]
    maxZVal = maxZVal - offset
    
    if maxZVal > 7:
        maxZVal = 6.5
    elif maxZVal < 4:
        maxZVal = 3.5
        
    print "Height Threshold : %f " % maxZVal
    
    return maxZVal 


def height_mask(project_dsm,thresh):
    """
    Load project_dsm and create height mask
    """
    mask_height = project_dsm > thresh
    mask_height = mask_height.astype(int)
    mask_height = cv2.convertScaleAbs(mask_height)
    
    return mask_height
    

def mask_add(mask_height,mask_rgb):
    """
    Combine color and height mask, denoise and return new masked project
    """
    
    ## Combine the tree mask with height mask
    mask_combined = cv2.bitwise_and(mask_height,mask_rgb)
    # Remove noises using open operation (Erosion followed by Dilation) to remove noises outside of the objects.
    kernel = np.ones((20,20),np.uint8)
    mask_denoise = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    return mask_denoise,mask_combined

def filtering(thresh1,filter_window = 151):
    """
    Apply Median Filter

    """    
    median = cv2.medianBlur(thresh1,filter_window)
    
    return median

def create_contours(median):

    """
    Find contours

    """
    mask = median.copy()
    if MODE == 'EDGE':
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    elif MODE == 'LOCAL':
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     contours, _ = cv2.findContours(median, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours
#     _, contours, _ = cv2.findContours(median, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     return contours

def vis_imgprocess(rgbArray,mask_rgb,mask_height,mask_combined,mask_denoise,contours,index):

    """

    Gerenrates visualizations of the intermediate and final img processes

    """    

    #img_dummy = mask_denoise.copy()

    plt.figure(figsize=(12,12));
    plt.subplot(231), plt.imshow(rgbArray), plt.title('orthomosaic image') 
    plt.subplot(232), plt.imshow(mask_rgb,cmap='gray'), plt.title('tree mask')
    plt.subplot(233), plt.imshow(mask_height,cmap='gray'), plt.title('height mask')
    plt.subplot(234), plt.imshow(mask_combined,cmap='gray'), plt.title('combined mask')
    plt.subplot(235), plt.imshow(mask_denoise,cmap='gray'), plt.title('combined mask after denoising')

    
    # Tried the following to create individual contour, didn't work
    #cv2.drawContours(img_dummy, contours, index, (0, 255, 0), 3)
    cv2.drawContours(mask_denoise, contours, -1, (0, 255, 0), 3)
    plt.subplot(236), plt.imshow(mask_denoise), plt.title('Contour')
    plt.show()
    
    
def georef(geofile):
    """
    Get pixel width and height in meters

    """    
        
    geoTransf = geofile.GetGeoTransform()
    xOrigin = geoTransf[0]
    yOrigin = geoTransf[3]
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
    
def compute_area_2d(contours,index,pixW = 0.00406,pixH = 0.00406,s2rRatio = 1.302):

    """

    Compute area using 2d info

    """ 
    ##DICTIONARY COMPREHENSION EFFICIENT
    
    #c,index = max([(cv2.contourArea(v),i) for i,v in enumerate(contours)])
    
    
    sqM2sqF = 10.7639
    #area = cv2.contourArea(cv2.convexHull(contours[index]))*pixW*pixH*s2rRatio*sqM2sqF
    area = cv2.contourArea(contours[index])*pixW*pixH*s2rRatio*sqM2sqF
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
    perc_error = (prediction - target)/target
    print 'actual roof sqft is:', target
    print 'RMSE is:', RMSE
    print 'perc_error:', perc_error
    
    return target,RMSE,perc_error


    
def pipeline_height(dsm,orthom,cloud_file):
	"""
	 Pileline all processing functions together using height mask only
     filename: color_relief.tif
     dsm: project_dsm.tif
     now the order is apply height mask then rgb mask
     good to try a different order --TODO!!
     
	"""
	start_time = time.time()

	# Load project_dsm, orthmosaic and point cloud files
	project_dsm,geofile = read_projdsm(dsm)
	rgbArray = load_orthomosaic(orthom)
	maxZVal = adaptive_thresh(cloud_file)
    
	# Create masks and return masked project
	mask_rgb = hsv_mask(rgbArray)
	mask_height = height_mask(project_dsm,thresh = maxZVal)
	mask_denoise,mask_combined = mask_add(mask_height,mask_rgb)
	median = filtering(mask_denoise)
	# Find contours and return the largest one
	contours = create_contours(median) 
	c,index = max_contour(contours) 

	# Predict roof area
	pixW,pixH = georef(geofile) # get pixel width and height in meters
	areaEst = compute_area_2d(contours,index,pixW = pixW,pixH = pixH)
	#vis_imgprocess(rgbArray,mask_rgb,mask_height,mask_combined,mask_denoise,contours,index) # visualize

    ###
    ###The following lines of code adjust height threshold based on the estimated area of first round
    ###and re-estimate roof square feet
    
# 	if areaEst < 900:
# 		mask_height = height_mask(project_dsm,thresh = 3)
# 		mask_denoise,mask_combined = mask_add(mask_height,mask_rgb)
# 		median = filtering(mask_denoise)
# 		# Find contours and return the largest one
# 		contours = create_contours(median) 
# 		c,index = max_contour(contours) 

# 	# Predict roof area
# 		pixW,pixH = georef(geofile) # get pixel width and height in meters
# 		areaEst = compute_area_2d(contours,index,pixW = pixW,pixH = pixH)
# 		vis_imgprocess(rgbArray,mask_rgb,mask_height,mask_combined,mask_denoise,contours,index) # visualize

# 	elif areaEst > 10000:
# 		mask_height = height_mask(project_dsm,thresh = 8)
# 		mask_denoise,mask_combined = mask_add(mask_height,mask_rgb)
# 		median = filtering(mask_denoise)
# 		# Find contours and return the largest one
# 		contours = create_contours(median) 
# 		c,index = max_contour(contours) 

#	# Predict roof area
# 		pixW,pixH = georef(geofile) # get pixel width and height in meters
# 		areaEst = compute_area_2d(contours,index,pixW = pixW,pixH = pixH)
# 		vis_imgprocess(rgbArray,mask_rgb,mask_height,mask_combined,mask_denoise,contours,index) # visualize

        
	exe_time = time.time() - start_time
	print "estimated area square feet is using height mask only is \n", areaEst
	print "seconds ",exe_time
	return areaEst,exe_time,maxZVal
	
    
    
    
    
def main():  
    start_time = time.time()
    
    	
    if MODE == 'LOCAL':
    	rootdir = '/Users/ejlq/Documents/ARI-HackWeek/training/'
    	target_dict = create_groundTruth('ARI-targets.csv')
    elif MODE == 'EDGE':
    	rootdir = '/san-data/usecase/skyscout/ARI-HackWeek/training/'
    	target_dict = create_groundTruth('/home/ejlq/ARI-targets.csv')
    results = []
    
    files = [x for x in os.listdir(rootdir) if not x.startswith('.')]
    
    for f in files:
            
        result = []
        dsm = rootdir + f + '/dsm/project_dsm.tif'
        orthom = rootdir + f + '/orthomosaic_rgb/project_transparent_mosaic_rgb.tif'
        cloud_file = rootdir + f + '/point_cloud/project_densified_point_cloud.xyz'
        print '----computing the following house now:' , f

            
        #est, exe_time= pipeline_height(dsm,orthom,cloud_file)
        try:
        	est, exe_time,maxZVal= pipeline_height(dsm,orthom,cloud_file)
        except:
        	est, exe_time,maxZVal = 2700, 10000,6
        	print 'estimation error out'
        target,RMSE,perc_error = evaluate(target_dict,f,est)
            
        result.append(f)
        result.extend([est,target,exe_time,RMSE,perc_error,maxZVal])
        results.append(result)

    
    if MODE == 'LOCAL' :
       
    	with open("output2.csv", "wb") as f:
    		writer = csv.writer(f)
    		writer.writerow(['House','Predicted','Target','Exe_time','RMSE','Perc_error','maxZVal'])
    		writer.writerows(results)
    elif MODE == 'EDGE' :
        os.chdir('/home/ejlq/hackweek/output/')
    	with open("output.csv", "wb") as f:
    		writer = csv.writer(f)
    		writer.writerow(['House','Predicted','Target','Exe_time','RMSE','Perc_error','maxZVal'])
    		writer.writerows(results)
        
    print 'results saved to output.csv'
    print 'in total takes the followig minutes to run', (time.time() - start_time)/60
    return results
        
                

if __name__ == '__main__':
    main()