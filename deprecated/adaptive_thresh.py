import cv
import numpy
from scipy import signal

def thresh(a, b, max_value, C):
    return max_value if a > b - C else 0

def mask(a,b):
    return a if b > 100 else 0

def unmask(a,b,c):
    return b if c > 100 else a

v_unmask = numpy.vectorize(unmask)
v_mask = numpy.vectorize(mask)
v_thresh = numpy.vectorize(thresh)

def block_size(size):
    block = numpy.ones((size, size), dtype='d')
    block[(size - 1 ) / 2, (size - 1 ) / 2] = 0
    return block
    
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

def get_number_neighbours(mask,block):
    '''returns number of unmasked neighbours of every element within block'''
    mask = mask / 255.0
    return signal.convolve2d(mask, block, mode='same', boundary='symm')

def masked_adaptive_threshold(image,mask,max_value,size,C):
    '''thresholds only using the unmasked elements'''
    block = block_size(size)
    conv = signal.convolve2d(image, block, mode='same', boundary='symm')
    mean_conv = conv / get_number_neighbours(mask,block)
    return v_thresh(image, mean_conv, max_value,C)

image = cv.LoadImageM("image.png", cv.CV_LOAD_IMAGE_GRAYSCALE)
mask = cv.LoadImageM("mask.png", cv.CV_LOAD_IMAGE_GRAYSCALE)

#change the images to numpy arrays
original_image = numpy.asarray(image)
mask = numpy.asarray(mask)
# Masks the image, by removing all masked pixels.
# Elements for mask > 100, will be processed
image = v_mask(original_image, mask)
# convolution parameters, size and C are crucial. See discussion in link below.
image = masked_adaptive_threshold(image,mask,max_value=255,size=7,C=5)
# puts the original masked off region of the image back
image = v_unmask(original_image, image, mask)
#change to suitable type for opencv
image = image.astype(numpy.uint8)
#convert back to cvmat
image = cv.fromarray(image)

image = resize(image)

cv.ShowImage('image', image)
#cv.SaveImage('final.png',image)
cv.WaitKey(0)