import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np 
import cv2 

from PIL import Image , ImageOps

filename = '/home/i880/Pictures/Camera/pic.jpeg'
filename2 = '/home/i880/Pictures/Camera/pic2.jpg'
# image0 = mpimg.imread(filename)
# image0 = image0.flatten()
# image0 = Image.open(filename)
# image0 = ImageOps.grayscale(image0)
image0 = cv2.imread(filename)
image1 = cv2.imread(filename)
image0 = np.array(image0)
image1 = np.array(image1)

image0=cv2.resize(image0,(460,640))
image1=cv2.resize(image1,(460,640))
image3= image0 + image1
cv2.imshow("image0",image0)
cv2.imshow("image1",image1)
cv2.imshow("image3",image3)
cv2.waitKey(0)