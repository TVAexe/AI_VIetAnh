from yolov10.ultralytics import YOLOv10

MODEL_PATH = './yolov10/yolov10n.pt'
model = YOLOv10(MODEL_PATH)
print(model.info()) 

import locale
locale.getpreferredencoding = lambda: "UTF-8"
import matplotlib.pyplot as plt
import cv2
import numpy as np

IMAGE_URL = 'https://static.independent.co.uk/s3fs-public/thumbnails/image/2018/05/11/10/hanoi-main.jpg'
CONF_THRESHOLD = 0.5
IMG_SIZE = 640
results = model.predict(source=IMAGE_URL)
                    #    imgsz=IMG_SIZE,
                    #    conf=CONF_THRESHOLD)
annotated_img = results[0].plot()

# show ảnh đã dự đoán
# plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

# lưu ảnh đã dự đoán
cv2.imwrite('annotated_image.jpg', annotated_img)


