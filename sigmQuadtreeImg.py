import cv2 
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import os
#import random
import numpy as np
import math 



def printI(img):
    # use just opencv not matplotlib
    # set window
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cv2.imshow("image",img)
    cv2.waitKey(0)

# feed the img to function printI
#printI(img)

# make class node 
class Node():
    def __init__(self, x0,y0,w,h):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.children = []

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_points1(self):
        return self.points

    def get_points(self,img):
        return img[self.x0:self.x0 + self.get_width(),self.y0:self.y0 + self.get_height()]

    def get_error(self,img):
        pixels = self.get_points(img)
        b_avg = np.mean(pixels[:,:,0])
        b_mse = np.square(np.subtract(pixels[:,:,0],b_avg)).mean()

        g_avg = np.mean(pixels[:,:,1])
        g_mse = np.square(np.subtract(pixels[:,:,1],g_avg)).mean()
        
        r_avg = np.mean(pixels[:,:,2])
        r_mse = np.square(np.subtract(pixels[:,:,2],r_avg)).mean()
        
        e = r_mse*0.2989 + g_mse*0.5870 + b_mse*0.1140
        return (e * img.shape[0]*img.shape[1])/ 90000000

        

#feed img to class Node
#node = Node(34,48,img.shape[1],img.shape[0])
#print("error = ", node.get_error(img))
# output error =  2.854974739807691



class QTree():
    def __init__(self,stdThreshold,minPixelSize,img):
        self.stdThreshold = stdThreshold
        self.min_size = stdThreshold
        self.minPixelSize = minPixelSize
        self.img = img
        self.root = Node(0,0,img.shape[0],img.shape[1])

    def get_points(self):
        return self.img[self.root.x0:self.root.x0 + self.root.get_width(),self.root.y0:self.y0 + self.root.get_height()]

    def subdivide(self):
        recursive_subdivide(self.root,self.stdThreshold,self.minPixelSize,self.img)


    #def graph_tree(self):
    #   fig = plt.figure(figsize=(10,10))
    #   plt.title("Quad Tree")
    #   c = find_children(self.root)
    #   print("Number of segments = %d" %len(c))
    #   for n in c:
    #       plt.gcf().gca().add_patch(patches.Rectangle((n.y0,n.x0),n.get_height(),n.get_height(),linewidth=1,edgecolor='r',facecolor='none',fill=False))

    #   plt.gcf().gca().set_xlim(0, img.shape[1])
    #   plt.gcf().gca().set_ylim( img.shape[0],0)
    #    plt.axis('equal')
    #   plt.show()
        #return

    


    def graph_tree(self, img):

    # Make a copy of the original image to draw rectangles on it
        img_copy = img.copy()

    # Get the children (segments) from the quadtree
        children = find_children(self.root)
        print("Number of segments = %d" % len(children))

    # Loop through each node in the quadtree and draw its bounding box (rectangle)
        for node in children:

        # Draw a rectangle on the image
            top_left = (node.x0, node.y0)
            bottom_right = (node.x0 + node.width, node.y0 + node.height)
            cv2.rectangle(img_copy, top_left, bottom_right, color=(0, 0, 255), thickness=2)  # Red color, thickness=2

    # Display the image with rectangles
        cv2.namedWindow("Quad Tree", cv2.WINDOW_NORMAL)
        cv2.imshow("Quad Tree", img_copy)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()




    def render_img(self,thickness = 1,color = (0,0,255)):
        imgc = self.img.copy()
        c = find_children(self.root)
        for n in find_children(self.root):
            pixels = n.get_points(self.img)
            gAvg = math.floor(np.mean(pixels[:,:,0]))
            rAvg = math.floor(np.mean(pixels[:,:,1]))
            bAvg = math.floor(np.mean(pixels[:,:,2]))

            imgc[n.x0:n.x0 + n.get_width(),n.y0:n.y0 + n.get_height(),0] = gAvg
            imgc[n.x0:n.x0 + n.get_width(),n.y0:n.y0 + n.get_height(),1] = rAvg
            imgc[n.x0:n.x0 + n.get_width(),n.y0:n.y0 + n.get_height(),2] = bAvg

        if thickness >0:
            for n in c :
                  imgc = cv2.rectangle(imgc,(n.y0,n.x0),(n.y0+n.get_height(),n.x0+n.get_width()),color,thickness)
        return imgc

def recursive_subdivide(node,stdThreshold,minPixelSize,img):

    if node.get_error(img) < stdThreshold :
        return
    w_1 = int(math.floor(node.width/2))
    w_2 = int(math.ceil(node.width/2))
    h_1 = int(math.floor(node.height/2))
    h_2 = int(math.ceil(node.height/2))
    
    if w_1 <= minPixelSize or h_1 <= minPixelSize:
        return
    #top left
    x1 = Node(node.x0,node.y0,w_1,h_1)
    recursive_subdivide(x1,stdThreshold,minPixelSize,img)
    # btm left
    x2 = Node(node.x0,node.y0 + h_1,w_1,h_2)
    recursive_subdivide(x2,stdThreshold,minPixelSize,img)
    #top right
    x3 = Node(node.x0 + w_1,node.y0,w_2,h_1)
    recursive_subdivide(x3,stdThreshold,minPixelSize,img)
    #btm right
    x4 = Node(node.x0 + w_1,node.y0 + h_1,w_2,h_2)
    recursive_subdivide(x4,stdThreshold,minPixelSize,img)

    node.children = [x1,x2,x3,x4]


def find_children(node):
    if not node.children:
        return [node]
    else:
        children = []
        for child in node.children:
            children += find_children(child)
    return children
   


#concate two images original and segmented 
def concat_images(img1, img2,boarder=5,color=(255,255,255)):
    img1_boarder = cv2.copyMakeBorder(img1,
                                      boarder,
                                      boarder,
                                      boarder,
                                      boarder,
                                      cv2.BORDER_CONSTANT,
                                      value=color
                                      )
    img2_boarder = cv2.copyMakeBorder(img2,
                                      boarder,
                                      boarder,
                                      0,
                                      boarder,
                                      cv2.BORDER_CONSTANT,
                                      value=color
                                      )
    return np.concatenate((img1_boarder, img2_boarder),axis=1)

#display quadtree 
def displayQuadTree(img_name,threshold=7,minCell=1,img_boarder=20,line_boarder=1,line_color=(0,0,255)):
    imgT = cv2.imread(img_name)
    qt = QTree(img=imgT,stdThreshold=threshold,minPixelSize=minCell)
    qt.subdivide()
    qtImg = qt.render_img(thickness=line_boarder,color=line_color)

        # Ensure output directories exist
    os.makedirs("qtSegm_img", exist_ok=True)
    
    # Save the segmented image
    file_name = "qtSegm_img/" + os.path.basename(img_name)
    cv2.imwrite(file_name, qtImg)
    
    # Concatenate the original and segmented images
    hConcat = concat_images(imgT, qtImg, boarder=img_boarder, color=(255, 255, 255))
    
    # Save the concatenated result
    file_name2 = f"qtSegm_img/disptych-{os.path.basename(img_name)}"
    cv2.imwrite(file_name2, hConcat)
    
    # Display the concatenated result
    printI(hConcat)


    '''
    file_name = "qtSegm_img/"+ img_name.split("/")[-1]
    cv2.imwrite(file_name,qtImg)
    file_name2 = "qtSegm/disptych-" + img_name[-6] + img_name[-5] + ".jpeg"
    hConcat = concat_images(img,qtImg,boarder=img_boarder,color=(255,255,255))
    cv2.imwrite(file_name2,hConcat)
    printI(hConcat)
    '''


displayQuadTree(img_name="woman.jpeg",threshold=0.01,img_boarder=20,line_boarder=1,line_color=(0,0,255))

