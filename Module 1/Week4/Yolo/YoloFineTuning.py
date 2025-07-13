
from yolov10.ultralytics import YOLOv10
MODEL_PATH='./yolov10/yolov10n.pt'
model=YOLOv10(MODEL_PATH)
#print(model.info())

YAML_PATH='./custom_datasets/data.yaml'

EPOCHS=1000
IMG_SIZE=640
BATCH_SIZE=16

model.train(data=YAML_PATH,
epochs=EPOCHS,
batch=BATCH_SIZE,
imgsz=IMG_SIZE,
device='cuda')

TRAINED_MODEL_PATH='./yolov10/runs/detect/train/weights/best.pt'
model=YOLOv10(TRAINED_MODEL_PATH)

model.val(data=YAML_PATH,
imgsz=IMG_SIZE,
split='test')
