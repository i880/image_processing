import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
from scipy import signal

imgor = cv2.imread("woman.jpeg")
img = cv2.cvtColor(imgor, cv2.COLOR_BGR2RGB)
img = np.array(img)
# img =  img[:,:,0]*0.114 +  img[:,:,1]*0.587 +  img[:,:,2]*0.299  # BGR to grayscale
# img = np.where(img > 150, 255, 0).astype(np.uint8)  # threshold

# M = np.array([[0.412453, 0.357580, 0.180423],
#               [0.212671, 0.715160, 0.072169],
#               [0.019334, 0.119193, 0.950227]])  # RGB to XYZ
# img = np.dot(img, M.T)

# (x,y,z, min) = (1-img[:,:,2]/255, 1-img[:,:,1]/255, 1-img[:,:,0]/255, 0) 
# min = np.minimum(x, np.minimum(y, z))
# (c,m,y) = ((x-min)/(1-min)*255, (y-min)/(1-min)*255, (z-min)/(1-min)*255) 
# img[:,:,0], img[:, :, 1], img[:, :, 2] = c, m, y  # bgr to cmy

# lowpass filter
# kernel = np.array([[0.1, 0.1, 0.1],
#                    [0.1, 2/10, 0.1],
#                    [0.1, 0.1, 0.1]])
# img_lpf = cv2.filter2D(img, -1, kernel)
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
img_lpf = cv2.GaussianBlur(gray,(5,5),0)
kernel2 = np.array([[-1,-1,0],
                    [-1,0,1],
                    [0,1,1]])
img_hpf = cv2.filter2D(img_lpf, -1, kernel2)

# cv2.imshow("filtrph",img_hpf)
# cv2.waitKey(0)
# img_hpf = signal.convolve2d(img,kernel2)
# print(img)

# img_lpf = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(img_lpf,(5,5),0)
# img_hpf = cv2.Canny(blur,30,110)
# # img_hpf = cv2.Laplacian(blur,cv2.CV_16U)


def seuillage_70(img_arr):
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            if img_arr[i][j] > 70:
                img_arr[i][j] = 255
            else :
                img_arr[i][j] = 0
    return img_arr

def seuillage_25(img_arr):
    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            if img_arr[i][j] > 10:
                img_arr[i][j] = 255
            else :
                img_arr[i][j] = 0
    return img_arr
img_lpf70  = seuillage_70(img_hpf)
img_lpf25  = seuillage_25(img_hpf)
# kernel = np.ones((5,5),np.uint8)
# close = cv2.morphologyEx(img_lpf25,cv2.MORPH_CLOSE,kernel)

cv2.imshow("25", img_lpf25) 
cv2.imshow("70", img_lpf70)  
# cv2.imshow("close", close)  
cv2.waitKey(0)