# yolo_test.py
from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  # your weights file

img_path = r"data/images/train/helmet1.jpg"
img = cv2.imread(img_path)
if img is None:
    raise SystemExit(f"Failed to load {img_path}")

results = model(img)            # run inference
res0 = results[0]               # first image results

# boxes: xyxy, confidences, classes
boxes = res0.boxes
if boxes is None or len(boxes) == 0:
    print("No boxes detected.")
else:
    xyxy = boxes.xyxy.cpu().numpy().astype(int)  # shape (N,4)
    confs = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    for i, (b, c, cl) in enumerate(zip(xyxy, confs, cls), start=1):
        x1,y1,x2,y2 = b
        print(f"Box {i}: [{x1},{y1},{x2},{y2}] conf={c:.2f} cls={int(cl)}")
