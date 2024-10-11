# from PIL import Image
# import matplotlib.pyplot as plt 
# import seaborn as sb 
# import numpy as np 
# import cv2
# import pandas as pd
# path = "/home/i880/Pictures/Camera/pic2.jpg"
# path2 ="/home/i880/Pictures/Camera/pic.jpeg"
# image = cv2.imread(path)
# image = cv2.resize(image,(480,640))
# hist = cv2.calcHist(image,[0],None,[256],[0,256])
# plt.figure()
# plt.plot(hist,label="hsitogam")
# cv2.imshow("image",image)

# cv2.waitKey(0)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def compute_cumulative_histogram(image):
    # Flatten the image to get the pixel values
    pixel_values = image.flatten()
    # Calculate the histogram
    hist, bins = np.histogram(pixel_values, bins=256, range=(0, 256))
    # Calculate the cumulative histogram
    cum_hist = np.cumsum(hist)
    # Normalize the cumulative histogram
    cum_hist = cum_hist / cum_hist.max()
    return cum_hist

def compute_histogram_distance(hist1, hist2):
    # Calculate the Euclidean distance between the two histograms
    distance = np.linalg.norm(hist1 - hist2)
    return distance

# Load the images
image1 = mpimg.imread('/home/i880/Pictures/Camera/pic2.jpg')
image2 = mpimg.imread('/home/i880/Pictures/Camera/pic.jpeg')

# Convert to grayscale if the images are in RGB
if image1.ndim == 3:
    image1 = np.dot(image1[..., :3], [0.2989, 0.5870, 0.1140])
if image2.ndim == 3:
    image2 = np.dot(image2[..., :3], [0.2989, 0.5870, 0.1140])

# Compute the cumulative histograms
cum_hist1 = compute_cumulative_histogram(image1)
cum_hist2 = compute_cumulative_histogram(image2)

# Compute the distance between the cumulative histograms
distance = compute_histogram_distance(cum_hist1, cum_hist2)

# Plot the cumulative histograms
plt.figure()
plt.plot(cum_hist1, label='Image 1')
plt.plot(cum_hist2, label='Image 2')
plt.title('Cumulative Histograms')
plt.xlabel('Pixel Value')
plt.ylabel('Cumulative Frequency')
plt.legend()
plt.show()

print(f'The distance between the images is: {distance}')
