import numpy as np
import cv2

def processGrayImage(imagePath):
    image = cv2.imread(imagePath, 0)

    A=np.array([[1,0],[0,-1]])
    
    h=image.shape[0]
    w=image.shape[1]
    B=np.array([0,h-1])
    output = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            x1,x2= np.dot(A, np.array([j, i])) + B
            output[x2, x1] = image[i, j]
    cv2.imwrite('output_gray.png', output)


def processColorImage(imagePath):
    image = cv2.imread(imagePath, 1)

    image=image.astype(float)
    height,width,depth=image.shape
    A=np.array([[1,0],[0,-1]])
    B=np.array([0,height-1])
    output = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            pixel=image[i,j,:]
            new_j,new_i = np.dot(A, np.array([j, i])) + B
            output[new_i, new_j, :] = pixel
    output = output.astype(np.uint8)
    cv2.imwrite('output_color.png', output)

def processGrayImage2(imagePath):
    image = cv2.imread(imagePath, 0)

    A = np.array([[-1, 0], [0, 1]])
    h, w = image.shape
    B = np.array([w - 1, 0])
    output = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            x1, x2 = np.dot(A, np.array([j, i])) + B
            output[x2, x1] = image[i, j]
    
    cv2.imwrite('output_gray2.png', output)

def processColorImage2(imagePath):
    image = cv2.imread(imagePath, 1)

    image = image.astype(float)
    height, width, depth = image.shape
    A = np.array([[-1, 0], [0, 1]])
    B = np.array([width - 1, 0])
    output = np.zeros_like(image)
    
    for i in range(height):
        for j in range(width):
            pixel = image[i, j, :]
            new_j, new_i = np.dot(A, np.array([j, i])) + B
            output[new_i, new_j, :] = pixel
    
    output = output.astype(np.uint8)
    cv2.imwrite('output_color2.png', output)
    

processGrayImage('gray_image.png')
processColorImage('image.png')
processGrayImage2('gray_image.png')
processColorImage2('image.png')