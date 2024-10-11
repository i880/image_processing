import numpy as np 

import matplotlib.pyplot as plt 
import cv2

img1= cv2.imread("woman.jpeg")
img2 = cv2.imread("rgbimg.png")


img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

hist1, bins1 = np.histogram(img1.ravel(), 256, [0, 256])
hist2, bins2 = np.histogram(img2.ravel(), 256, [0, 256])

cdf1 = hist1.cumsum()
cdf2 = hist2.cumsum()

cdf1 = cdf1 / cdf1.max()
cdf2 = cdf2 / cdf2.max()

lut = np.interp(cdf1, cdf2, bins2[:len(cdf1)]).astype(np.uint8)

img1 = cv2.LUT(img1, lut)

hist1_eq, bins1_eq = np.histogram(img1.ravel(), 256, [0, 256])

plt.plot(hist1_eq, color='b')
plt.plot(hist2, color='r')
plt.show()

