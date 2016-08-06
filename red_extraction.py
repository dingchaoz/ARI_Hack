import cv2
import numpy as np
import os

os.chdir(r'/Users/ejlq/Documents/ARI-HackWeek/data/')


file_name = 'dsm/color_relief.tif'
img = cv2.imread(file_name)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_red = np.array([0,100,0])
upper_red = np.array([5,255,255])
mask = cv2.inRange(hsv,lower_red,upper_red)
res = cv2.bitwise_and(img,img,mask=mask)
#cv2.imshow('MASKING',mask)
cv2.imshow('ONLY_RED_COLOR_PASSED',res)

cv2.waitKey(0)
cv2.destroyAllWindows()