import numpy as np 
import cv2 

img = cv2.imread("woman.jpeg")
def quadtree_segmentation(img, threshold, max_depth=5):
    if max_depth == 0:
        return [img]
    else:
        width, height, _ = img.shape
        width2 = width // 2
        height2 = height // 2
        img1 = img[0:width2, 0:height2]
        img2 = img[width2:width, 0:height2]
        img3 = img[0:width2, height2:height]
        img4 = img[width2:width, height2:height]

        img1_mean = np.mean(img1)
        img2_mean = np.mean(img2)
        img3_mean = np.mean(img3)
        img4_mean = np.mean(img4)
        if np.abs(img1_mean - img2_mean) > threshold and np.abs(img1_mean - img3_mean) > threshold and np.abs(img2_mean - img3_mean) > threshold and np.abs(img3_mean - img4_mean) > threshold:
            return quadtree_segmentation(img1, threshold, max_depth - 1) + quadtree_segmentation(img2, threshold, max_depth - 1) + quadtree_segmentation(img3, threshold, max_depth - 1) + quadtree_segmentation(img4, threshold, max_depth - 1)
        else:
            return [img]

img_segments = quadtree_segmentation(img, 0.5, 5)


cv2.waitKey(0)