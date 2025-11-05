# detect_and_crop.py
from ultralytics import YOLO
import cv2
import pytesseract
import os
import csv

# ✅ PATH TO YOUR TRAINED MODEL
MODEL_PATH = r"runs/detect/helmet_numberplate_detection/weights/best.pt"

# ✅ PATH TO YOUR TEST IMAGES
IMAGE_DIR = r"data/images/val"

# ✅ FOLDER TO SAVE CROPPED NUMBER PLATES
CROPPED_DIR = "cropped_plates"
os.makedirs(CROPPED_DIR, exist_ok=True)

# ✅ CSV FILE TO SAVE OCR RESULTS
CSV_PATH = os.path.join(CROPPED_DIR, "plate_results.csv")

# ✅ TESSERACT CONFIG (if installed manually, add full path)
# Example (uncomment if needed):
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ LOAD TRAINED YOLO MODEL
model = YOLO(MODEL_PATH)

# ✅ OPEN CSV FILE TO SAVE OUTPUT
with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "Detected_Class", "Extracted_Text"])

    # ✅ LOOP THROUGH EACH IMAGE IN FOLDER
    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(IMAGE_DIR, img_name)
        print(f"\n🔍 Processing: {img_name}")

        # ✅ RUN DETECTION
        results = model(img_path)

        for result in results:
            boxes = result.boxes
            cls_names = result.names

            img = cv2.imread(img_path)

            for box in boxes:
                cls_id = int(box.cls[0])
                label = cls_names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ✅ CROP AND SAVE NUMBER PLATE IMAGE
                if label == "number_plate":
                    crop = img[y1:y2, x1:x2]
                    crop_filename = f"{os.path.splitext(img_name)[0]}_number_plate.jpg"
                    crop_path = os.path.join(CROPPED_DIR, crop_filename)
                    cv2.imwrite(crop_path, crop)

                    # ✅ OCR (Extract text from number plate)
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    extracted_text = pytesseract.image_to_string(
                        gray, config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    ).strip()

                    writer.writerow([img_name, label, extracted_text])
                    print(f"🪪 Extracted Text: {extracted_text}")

                # ✅ LOG helmet / no_helmet detections
                elif label in ["helmet", "no_helmet"]:
                    writer.writerow([img_name, label, ""])
                    print(f"🪖 Detected: {label} ({conf:.2f})")

print(f"\n✅ All tasks completed successfully!")
print(f"📁 OCR results saved in: {CSV_PATH}")