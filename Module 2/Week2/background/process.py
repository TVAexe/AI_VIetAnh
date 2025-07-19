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
    _, difference = cv2.threshold(difference, 15, 255, cv2.THRESH_BINARY)
    
    output=np.where(difference ==0, fakebackground, stillImage)
    cv2.imwrite('output.png', output)
backgroundExtract('StillImage.png', 'background.png', 'FakeBackground.png')