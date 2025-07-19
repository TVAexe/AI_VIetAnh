import numpy as np
import cv2

def backgroundExtract(stillImage,background,fakebackground):
    stillImage = cv2.imread(stillImage, 1)
    background = cv2.imread(background, 1)
    fakebackground = cv2.imread(fakebackground, 1)

    stillImage=cv2.resize(stillImage, (640, 480))
    background=cv2.resize(background, (640, 480))
    fakebackground=cv2.resize(fakebackground, (640, 480))

    difference= cv2.absdiff(stillImage, background)
    difference_single=np.sum(difference, axis=2)/3.0
    difference_single = difference_single.astype(np.uint8)
    difference_binary=np.where(difference_single >=15, 255, 0).astype(np.uint8)
    difference_binary = np.stack((difference_binary,)*3, axis=-1)
    output = np.where(difference_binary == 255, stillImage, fakebackground)
    cv2.imwrite('output.png', output)
backgroundExtract('StillImage.png', 'background.png', 'FakeBackground.png')