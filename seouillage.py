import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('woman.jpeg')
img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

kernel = np.array([1,0,-1])
IGx = cv2.filter2D(img,-1,kernel)

IGy = cv2.filter2D(img,-1,kernel.T)

IG = np.abs(IGx) + np.abs(IGy)
IG6 = np.where(IG > 6, 255, 0).astype(np.uint8)
IG5 = np.where(IG > 5, 255, 0).astype(np.uint8)

_, IGt = cv2.threshold(IG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
IGh = cv2.Canny(IG, 6, 7, apertureSize=3)
IGh = cv2.bitwise_or(IGt, IGh)

cv2.namedWindow('IGh', cv2.WINDOW_NORMAL)
cv2.resizeWindow('IGh', 640, 480)
IGh = cv2.resize(IGh.astype(np.uint8),(640,480))
cv2.imshow('IGh',IGh)

cv2.namedWindow('IG', cv2.WINDOW_NORMAL)
cv2.resizeWindow('IG', 640, 480)
IG = cv2.resize(IG.astype(np.uint8),(640,480))
cv2.imshow('IG',IG)
cv2.waitKey(0)
