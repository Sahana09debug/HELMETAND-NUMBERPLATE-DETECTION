import os
import cv2
import pytesseract
import csv
import numpy as np
from ultralytics import YOLO

# ✅ Path to tesseract.exe (Update if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------------------------------------------------
# 🧠 Preprocessing for OCR
# -------------------------------------------------------------------
def preprocess_plate(plate_img):
    """Enhanced preprocessing for better OCR accuracy."""
    if plate_img is None or plate_img.size == 0:
        return None

    # Resize small plates for better recognition
    height, width = plate_img.shape[:2]
    if height < 100 or width < 300:
        scale = max(3, 300 / width, 100 / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        plate_img = cv2.resize(plate_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img

    # Denoise using bilateral filter
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)

    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    # Thresholding
    _, thresh1 = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh2 = cv2.adaptiveThreshold(contrast_enhanced, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    _, thresh3 = cv2.threshold(contrast_enhanced, 120, 255, cv2.THRESH_BINARY)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned1 = cv2.morphologyEx(cleaned1, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel, iterations=1)

    return {
        'original': plate_img,
        'gray': gray,
        'enhanced': contrast_enhanced,
        'otsu': cleaned1,
        'adaptive': cleaned2,
        'binary': thresh3
    }

# -------------------------------------------------------------------
# 🔍 OCR Text Extraction
# -------------------------------------------------------------------
def extract_text_multiple_methods(cropped_plate, cropped_path):
    """Try multiple OCR configs and preprocessing methods."""
    if cropped_plate is None or cropped_plate.size == 0:
        return "❌ Invalid image"

    processed_images = preprocess_plate(cropped_plate)
    if processed_images is None:
        return "❌ Preprocessing failed"

    # Save debug versions
    debug_folder = os.path.join(os.path.dirname(cropped_path), "debug")
    os.makedirs(debug_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(cropped_path))[0]

    for method_name, img in processed_images.items():
        debug_path = os.path.join(debug_folder, f"{base_name}_{method_name}.jpg")
        cv2.imwrite(debug_path, img)

    configs = [
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',        
    ]

    results = []

    for method_name, processed_img in processed_images.items():
        if method_name == 'original':
            continue
        for config in configs:
            try:
                text = pytesseract.image_to_string(processed_img, config=config)
                text = text.strip().replace(' ', '').replace('\n', '').replace('-', '')
                text = ''.join(c for c in text if c.isalnum()).upper()

                if len(text) >= 4:
                    results.append((text, method_name, config))
            except Exception:
                continue

    if results:
        results.sort(key=lambda x: len(x[0]), reverse=True)
        best_text, best_method, best_config = results[0]
        print(f"   ✓ Best method: {best_method} | Text: {best_text}")
        return best_text

    return "❌ No text detected"

# -------------------------------------------------------------------
# ⚙ YOLO Model Load
# -------------------------------------------------------------------
model = YOLO(r"C:\Users\dell\OneDrive\Desktop\HelmetNumberplate\runs\detect\helmet_numberplate_detection\weights\best.pt")

# -------------------------------------------------------------------
# 📁 Paths
# -------------------------------------------------------------------
image_folder = r"data\images\val"             # Input images
output_folder = "cropped_plates"
processed_folder = os.path.join(output_folder, "processed")
csv_path = os.path.join(output_folder, "plate_results.csv")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)

# -------------------------------------------------------------------
# 🚀 Detection + OCR Pipeline
# -------------------------------------------------------------------
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

                # 🧾 Number plate detection
                if "plate" in label.lower():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    img = cv2.imread(img_path)

                    # Add padding to ensure full plate is captured
                    padding = 20
                    h, w = img.shape[:2]
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)

                    cropped = img[y1:y2, x1:x2]
                    cropped_name = f"plate_{img_name}"
                    cropped_path = os.path.join(processed_folder, cropped_name)
                    cv2.imwrite(cropped_path, cropped)

                    print(f"   📸 Plate size: {cropped.shape}")
                    ocr_result = extract_text_multiple_methods(cropped, cropped_path)

                    if "❌" not in ocr_result:
                        print(f"   🪪 Extracted Text: {ocr_result}")
                    else:
                        print(f"   {ocr_result}")

                    writer.writerow([img_name, label, round(conf, 2), ocr_result])

                # 🪖 Helmet detections
                elif label in ["helmet", "no_helmet"]:
                    print(f"🪖 Detected: {label} ({round(conf, 2)})")
                    writer.writerow([img_name, label, round(conf, 2), "N/A"])

print("\n✅ All tasks completed successfully!")
print(f"📁 OCR results saved at: {csv_path}")
print(f"📸 Processed plate images saved at: {processed_folder}")
print(f"🔧 Debug preprocessed images saved at: {os.path.join(processed_folder, 'debug')}")
