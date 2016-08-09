import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(r'/Users/ejlq/Documents/ARI-HackWeek/data/')


file_name = 'dsm/color_relief.tif'
img = cv2.imread(file_name)
#hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_red = np.array([15,114,209])
upper_red = np.array([200,200,253])
#upper_red = np.array([115,148,253])
mask = cv2.inRange(img,lower_red,upper_red)
output = cv2.bitwise_and(img,img,mask=mask)
#cv2.imshow('MASKING',mask)

# we need to keep in mind aspect ratio so the image does
# not look skewed or distorted -- therefore, we calculate
#the ratio of the new image to the old image
r = 100.0 / output.shape[1]
dim = (100, int(output.shape[0] * r))

# perform the actual resizing of the image and show it
resized = cv2.resize(output, dim, interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY)
median = cv2.medianBlur(thresh1,15)
#cv2.imshow('Median Blur',median)
#edges = cv2.Canny(thresh1,10,20)


im2, contours, hierarchy = cv2.findContours(median, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# img_dummy = median.copy()
# cv2.drawContours(img_dummy, contours, -1, (0, 255, 0), 3)
# # 
# plt.imshow(img_dummy),plt.title('Contour')
# plt.show()
# # 
area = 0
pixels = 0
count = 0
total_count = 0
index = []
for c in contours:
	#M = cv2.moments(c)
	M = cv2.contourArea(c)
	total_count ++1
	if M > 10:
		pixels = pixels + M
		count = count + 1
		index.append(c)
		img_dummy = median.copy()
		cv2.drawContours(img_dummy, [c], 0, (0, 255, 0), 3)
		plt.imshow(img_dummy),plt.title('Contour')
		plt.show()
	
	# print M	
# 	print "\n"
area = pixels * 1.6*10.7639
print "pixels is", pixels
print "number of contours is", count
print "the index of the countour which potentianly is the roof is", index
print "area square feet is ", area

# cv2.imshow("resized", median)
# cv2.waitKey(0)




# cv2.imshow('ONLY_RED_COLOR_PASSED',output)
# 
# cv2.waitKey(0)
# cv2.destroyAllWindows()