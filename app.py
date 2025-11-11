from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
import cv2
import pytesseract
import pandas as pd
from PIL import Image
import uuid  # to make unique folder names

app = Flask(__name__)

# Folder paths
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'runs/detect/helmet_numberplate_detection/weights/best.pt'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Check for uploaded file
    if 'file' not in request.files:
        return "No file uploaded!"

    file = request.files['file']
    if file.filename == '':
        return "No file selected!"

    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # ✅ Convert .avif to .jpg automatically
    if file_path.lower().endswith('.avif'):
        img = Image.open(file_path)
        new_path = file_path.rsplit('.', 1)[0] + '.jpg'
        img.save(new_path)
        os.remove(file_path)
        file_path = new_path

    # Generate unique folder name for results
    unique_folder = f"detect_{uuid.uuid4().hex[:8]}"

    # Run YOLO detection
    results = model(file_path, save=True, project=RESULT_FOLDER, name=unique_folder)

    # Get the output image path
    output_dir = os.path.join(RESULT_FOLDER, unique_folder)
    detected_files = os.listdir(output_dir)
    result_image_path = os.path.join(output_dir, detected_files[0]) if detected_files else file_path

    # Read image for OCR
    img = cv2.imread(file_path)
    detected_number = "No number detected"

    # OCR on number plate detections
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = results[0].names[cls]

        # If number plate detected, crop and read text
        if "number" in label.lower() or "plate" in label.lower():
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            detected_number = pytesseract.image_to_string(gray).strip()
            break

    # Save results to CSV
    csv_path = 'plate_results.csv'
    df = pd.DataFrame([[file.filename, detected_number]], columns=['Image', 'Detected Number'])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    # Return the result page
    return render_template('result.html', image=result_image_path, number=detected_number)

if __name__ == "__main__":
    app.run(debug=True)
