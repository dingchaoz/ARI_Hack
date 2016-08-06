"""
Author: Gary Foreman
Created: March 17, 2016
Script to generate border shape files from georefenced image data. Shape files
are used to mask out image regions that have pixel values of zero so that they
may be ignored in downstream applications to enhance computational efficiency.
"""

from __future__ import print_function
import argparse
import cv2                      #version 3.1
import gdal                     #part of gdal 1.11
from gdalconst import *         #part of gdal 1.11
import glob
import matplotlib.pyplot as plt
import numpy as np
import ogr                      #part of gdal 1.11
import os
import subprocess

#play with these parameters if you're having trouble with your contours
BOOL_CUT = 0.01
FILTER_WINDOW = 75 #must be an odd, positive integer

PATH_TO_OGR2OGR = '/Library/Frameworks/GDAL.framework/Versions/1.11/Programs/'
          
def create_contours(img_grey, bool_cut=10, filter_window=7):
    """
    img_grey: grey scale image loaded using cv2.imread
    bool_cut: integer, level above which to mark pixel as true, otherwise, false
    filter_window: odd integer, size of window to use for median filtering
    identifies contour of the img_grey outside of which pixel values are
    effectively zero.
    returns:
      contours: list of pixels that are contour vertices
      img_bool: original boolean image indicating values above and below
                bool_cut
      img_bool_median_filter: median filtered boolean image
    """
    
    img_bool = np.array(img_grey > bool_cut, dtype=np.uint8)
    img_bool_median_filter = cv2.medianBlur(img_bool, filter_window)
    
    _, contours, _ = cv2.findContours(img_bool_median_filter, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

    return contours, img_bool, img_bool_median_filter                                        
                                      
def plot_contours(img, img_grey, img_bool, img_bool_median_filter, contours):
    """
    img: rgb image loaded using cv2.imread
    img_grey: grey scale image loaded using cv2.imread
    img_bool: output of create_contours
    img_bool_median_filter: output of create_contours
    contours: output of create_contours
    generates visualization of image and contour data
    """
    
    img_dummy = img.copy()

    plt.subplot(231), plt.imshow(img), plt.title('Original')
    plt.subplot(232), plt.imshow(img_grey, cmap='gray'), plt.title('Grey')
    plt.subplot(234), plt.imshow(img_bool*255, cmap='gray'),
    plt.title('Boolean (Grey >10)')
    plt.subplot(235), plt.imshow(img_bool_median_filter*255, cmap='gray'),
    plt.title('Boolean (Median Filtered)')
    
    cv2.drawContours(img_dummy, contours, -1, (255, 0, 0), 3)
    plt.subplot(236), plt.imshow(img_dummy), plt.title('Contour')
    plt.show()
    
def xy2wgs84(img_gdal, contour_vertices):
    """
    img_gdal: image file loaded using gdal.Open
    contour_vertices: 2-d array, reshaped array of contours output by
                                 create_contours
    returns array of contour vertices in wgs84 coordinates
    """
    
    geo_transform = img_gdal.GetGeoTransform()
    return(map(lambda x: gdal.ApplyGeoTransform(geo_transform, float(x[0]),
                                                float(x[1])),
               contour_vertices))
               
def create_polygon(contours_wgs84):
    """
    contours_wgs84: contour vertices in wgs84 coordinates as output by
                    xy2wgs84
    returns an ogr.wkbPolygon 
    """
    
    #Add the points to the ring
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for point in contours_wgs84:
        ring.AddPoint(point[0], point[1])
        
    #Add first point again to ring to close polygon
    ring.AddPoint(contours_wgs84[0][0], contours_wgs84[0][1])
    
    #Add the ring to the polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    
    return(poly)
    
def write_to_esri(polygons, filename):
    """
    polygons: list of ogr.wkbPolygon as output by create_polygon
    filename: string, name of original image file
    writes polygon to esri shape file
    """
    
    path, ext = os.path.splitext(filename)
    outfile = path + '_border.shp'
    
    driver = ogr.GetDriverByName('ESRI Shapefile')
    
    # Remove output shapefile if it already exists
    if os.path.exists(outfile):
        driver.DeleteDataSource(outfile)
    
    #Create the output shapefile
    ds = driver.CreateDataSource(outfile)
    layer = ds.CreateLayer('border', geom_type=ogr.wkbPolygon)
    
    #Add an ID field
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    layer.CreateField(idField)
    
    #Create the feature and set values
    for i, polygon in enumerate(polygons):
        featureDefn = layer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(polygon)
        feature.SetField("id", i)
        layer.CreateFeature(feature)
    
    ds.Destroy()
    
    return()
    
def merge_shape_files(directory):
    """
    directory: string, in which to search for esri shape files
    merge all shape files created from individual images in directory into one
    master shape file
    """
    
    border_master = directory + '/border_master'
    
    #if boarder_master files already exist, remove them
    if os.path.exists(border_master + '.shp'):
        subprocess.call(['rm', border_master + '.shp', border_master + '.shx',
                         border_master + '.dbf'])
                         
    filelist = glob.glob(directory + '/*.shp')
    first_file = filelist.pop(0)
    first_path, ext = os.path.splitext(first_file)
    
    #create boarder_master files
    subprocess.call(['cp', first_path + '.shp', border_master + '.shp'])
    subprocess.call(['cp', first_path + '.shx', border_master + '.shx'])
    subprocess.call(['cp', first_path + '.dbf', border_master + '.dbf'])
    
    #perform merge over rest of filelist
    for filename in filelist:
        subprocess.call(['ogr2ogr', '-update', '-append',
                         border_master + '.shp', filename])

def handle_args():
    """
    command line arguments parser
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', action='store',
                        help='Name of I/O directory')
    parser.add_argument('-e', '--extension', action='store', default='jpg',
                        help='Extension of files to be read')
    args = parser.parse_args()
    
    if args.directory is None:
        print('Setting I/O directory to current working directory')
        args.directory = '.'
        
    return args

def main():
    args = handle_args()
    filelist = glob.glob(args.directory + '/*.' + args.extension)
    
    #these were troublesome files that I wanted to visualize
    #filelist = ['may24C357500e4102500n.jpg']
    #filelist = ['may26C370000e4105000n.jpg']
    #filelist = ['05OCT01171200-S2AS_R1C01-052272242030_01_P001.TIF']
    #filelist = ['05SEP26170642-S2AS_R1C06-052272242010_01_P001.TIF']
    #filelist = ['aug30C0884330w302230n.jpg']

    for infile in filelist:
        img = cv2.imread(infile)
        img_grey = cv2.imread(infile, 0)
        img_gdal = gdal.Open(infile, GA_ReadOnly)

        contours, img_bool, img_bool_median_filter = \
            create_contours(img_grey, bool_cut=BOOL_CUT,
                            filter_window=FILTER_WINDOW)
        
        #uncomment to generate plots
        #plot_contours(img, img_grey, img_bool, img_bool_median_filter,
        #              contours)

        polygons = [] #may have more than one polygon per image
        for i in xrange(len(contours)):
            contour_vertices = contours[i].reshape(len(contours[i]), 2)    
            contours_wgs84 = xy2wgs84(img_gdal, contour_vertices)
            polygons.append(create_polygon(contours_wgs84))
        write_to_esri(polygons, infile)
    
    merge_shape_files(args.directory)
    
if __name__ == '__main__':
    main()