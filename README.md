📌 Overview

This project detects:
Helmet usage (Helmet / No Helmet)
Vehicle number plates (for OCR recognition)

It uses:
YOLOv8 (Ultralytics) for object detection
Tesseract OCR for reading number plates
OpenCV for image preprocessing and visualization


Team Members and Responsibilities
| Member                               | Role                        | Main Tasks                                                                                    |
| :----------------------------------- | :-------------------------- | :-------------------------------------------------------------------------------------------- |
| **Member 1 (Data Engineer)**         | Dataset Preparation         | Gather images, annotate using LabelImg, denoise and preprocess dataset, create YOLO structure |
| **Member 2 (Model Trainer)**         | Model Training              | Train YOLOv8 on the dataset, tune hyperparameters, and save weights                           |
| **Member 3 (OCR Specialist)**        | Number Plate Recognition    | Implement and test OCR (Tesseract) to extract plate numbers from detected regions             |
| **Member 4 (Integration & Testing)** | Integration + Final Testing | Integrate YOLO + OCR pipeline, run predictions, and test on real videos/images                |

Step 1: Install Dependencies (All Members)
Each member should have Python 3.10+ installed.

Create a virtual environment
python -m venv venv
venv\Scripts\activate

Install required libraries
pip install ultralytics opencv-python pytesseract numpy matplotlib
If you face errors with Tesseract:

Download and install Tesseract from:
👉 https://github.com/UB-Mannheim/tesseract/wiki
Then note its path (for example):
C:\Program Files\Tesseract-OCR\tesseract.exe

Add this to your Python script later:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

tep 4: Install Required Packages
pip install -r requirements.txt


Your requirements.txt should include:

ultralytics==8.3.205
opencv-python
pillow
numpy
pandas
torch
tqdm
pytesseract

Step 2: Member 1 — Dataset Preparation
🔹 1. Folder structure
Create folders like this:
HelmetNumberplate/
│
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   └── labels/
│       ├── train/
│       ├── val/
├── scripts/
│   ├── noise_filter.py
│   ├── preprocess.py
├── dataset.yaml
└── runs/

Step 3: Member 2 — YOLOv8 Model Training
🔹 1. Create dataset.yaml
train: data/images/train
val: data/images/val

nc: 3
names: ['Helmet', 'NoHelmet', 'NumberPlate']

🔹 2. Train the model
yolo detect train data="dataset.yaml" model="yolov8n.pt" epochs=50 imgsz=640

This will create:
runs/detect/train/
└── weights/
    ├── best.pt
    └── last.pt

🔹 3. Validate model
yolo detect val model="runs/detect/train/weights/best.pt" data="dataset.yaml"

Step 4: Member 3 — Number Plate OCR
🔹 1. Import Tesseract
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

🔹 2. Apply OCR on cropped number plate
image = cv2.imread('car.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(gray)
print("Detected Text:", text)

🔹 3. Integrate OCR after YOLO detection

Use YOLO’s bounding box to crop the number plate region and pass it to pytesseract.

🧩 Step 5: Member 4 — Integration & Testing
🔹 1. Run predictions
yolo detect predict model="runs/detect/train/weights/best.pt" source="data/images/val"

🔹 2. Combine YOLO + OCR

In one Python file (main.py):

from ultralytics import YOLO
import cv2, pytesseract

model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict(source="test_images/car1.jpg", save=True)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = r.orig_img[y1:y2, x1:x2]
        text = pytesseract.image_to_string(crop)
        print("Detected Number:", text)


        
