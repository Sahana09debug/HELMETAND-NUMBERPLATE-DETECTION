# train_model.py
from ultralytics import YOLO

# ✅ Path to your dataset YAML file
DATA_YAML = r"data.yaml"

# ✅ Create or choose where to save model training results
MODEL_OUTPUT = r"runs/detect/helmet_numberplate_detection"

# ✅ Load a pre-trained YOLOv8 model (small & fast)
# You can also try 'yolov8m.pt' or 'yolov8l.pt' if you want more accuracy
model = YOLO("yolov8s.pt")

# ✅ Train the model
model.train(
    data=DATA_YAML,
    epochs=50,           # You can increase to 100+ if you have time
    imgsz=640,           # Image size for training
    batch=8,             # Adjust based on your GPU/CPU power
    name="helmet_numberplate_detection",
    project="runs/detect",
    workers=4,
    patience=20,         # Early stop if no improvement
    save=True,
    verbose=True
)

# ✅ Evaluate on validation set
metrics = model.val()

# ✅ Export trained model to weights folder
model.export(format="onnx")  # optional: exports for deployment

print("\n✅ Training Completed Successfully!")
print(f"📁 Model saved in: {MODEL_OUTPUT}/weights/best.pt")