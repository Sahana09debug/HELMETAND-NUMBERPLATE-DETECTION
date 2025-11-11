import os
import cv2
import csv
import re
from ultralytics import YOLO
import easyocr
import numpy as np

# -------------------------------------------------------------
# 🧠 Initialize EasyOCR
# -------------------------------------------------------------
reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a GPU

# -------------------------------------------------------------
# 🧹 Clean and validate extracted plate text
# -------------------------------------------------------------
def clean_plate_text(text):
    """Cleans OCR text and checks if it looks like a valid number plate."""
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)  # Keep only letters and digits
    if 6 <= len(text) <= 10:
        return text
    return "Invalid/Noisy text"

# -------------------------------------------------------------
# 🧾 YOLO model path
# -------------------------------------------------------------
MODEL_PATH = r"C:\Users\dell\OneDrive\Desktop\HelmetNumberplate\runs\detect\helmet_numberplate_detection\weights\best.pt"

# Load trained YOLO model
model = YOLO(MODEL_PATH)

# -------------------------------------------------------------
# 📁 Folders
# -------------------------------------------------------------
image_folder = r"data\images\val"         # Input images folder
output_folder = "cropped_plates"
processed_folder = os.path.join(output_folder, "processed")
csv_path = os.path.join(output_folder, "plate_results.csv")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)

# -------------------------------------------------------------
# ⚙ Detection + OCR Pipeline
# -------------------------------------------------------------
with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "Detected Object", "Confidence", "Extracted Text"])

    for img_name in os.listdir(image_folder):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        img_path = os.path.join(image_folder, img_name)
        print(f"\n🔍 Processing: {img_name}")

        results = model(img_path)

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                print("⚠ No detections found.")
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                # Detect number plate
                if "plate" in label.lower():
                    img = cv2.imread(img_path)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Add small padding
                    padding = 10
                    h, w = img.shape[:2]
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)

                    cropped = img[y1:y2, x1:x2]
                    cropped_name = f"plate_{img_name}"
                    cropped_path = os.path.join(processed_folder, cropped_name)
                    cv2.imwrite(cropped_path, cropped)
                    print(f"📸 Cropped plate saved: {cropped_path}")

                    # ------------------------------------------
                    # 🔠 EasyOCR Text Extraction
                    # ------------------------------------------
                    ocr_result = reader.readtext(cropped)

                    if ocr_result:
                        extracted_texts = [res[1] for res in ocr_result]
                        cleaned_results = [clean_plate_text(t) for t in extracted_texts]
                        valid_results = [t for t in cleaned_results if "Invalid" not in t]

                        if valid_results:
                            best_text = valid_results[0]
                        else:
                            best_text = cleaned_results[0]
                    else:
                        best_text = "❌ No text detected"

                    print(f"🪪 Extracted Text: {best_text}")
                    writer.writerow([img_name, label, round(conf, 2), best_text])

                # Helmet detections
                elif label.lower() in ["helmet", "no_helmet"]:
                    print(f"🪖 Detected: {label} ({round(conf, 2)})")
                    writer.writerow([img_name, label, round(conf, 2), "N/A"])

print("\n✅ All tasks completed successfully!")
print(f"📁 OCR results saved at: {csv_path}")
print(f"📸 Processed plate images saved at: {processed_folder}")