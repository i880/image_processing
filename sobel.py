import cv2
import numpy as np
img = cv2.cvtColor(cv2.imread("rgbimg.png"),cv2.COLOR_BGR2RGB)


img = img[:,:,0]
print()
cv2.imshow("img", img)
cv2.waitKey(0)