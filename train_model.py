# train_model.py
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# Load YOLOv8 model (choose small or nano)
model = YOLO('yolov8s.pt')  # or 'yolov8n.pt' for faster training

# Train the model
results = model.train(
    data='data.yaml',   # Path to data.yaml
    epochs=50,
    batch=16,
    imgsz=640,
    name='helmet_numberplate_detection'
)

# After training, show training results graph
img_path = os.path.join('runs', 'detect', 'helmet_numberplate_detection', 'results.png')
if os.path.exists(img_path):
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
else:
    print("Training results image not found.")
