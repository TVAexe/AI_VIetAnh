import cv2
import numpy as np
image = cv2.imread('image.png',1)

def color2gray(vector):
    result=vector.mean()
    result=result.astype(np.uint8)
    return result

def color2gray2(vector):
    result = np.dot(vector, [0.299, 0.587, 0.114])
    result = result.astype(np.uint8)
    return result
gray_image = np.apply_along_axis(color2gray, 2, image)
gray_image2 = np.apply_along_axis(color2gray2, 2, image)
cv2.imwrite('gray_image.png', gray_image)
cv2.imwrite('gray_image2.png', gray_image2)