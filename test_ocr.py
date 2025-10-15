# test_ocr.py

import cv2
import pytesseract
import os
import glob

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Path to the folder containing helmet images
image_folder = r"C:\Users\Thvishan\HELMETAND-NUMBERPLATE-DETECTION\data\images\denoised\train"

# Verify folder exists
if not os.path.exists(image_folder):
    print("Train folder not found at:", image_folder)
    exit()

# Find all helmet images automatically
image_paths = glob.glob(os.path.join(image_folder, "helmet*.jpg"))

if not image_paths:
    print("No helmet images found in folder!")
    exit()

# Loop through all helmet images
for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        continue

    # Resize image to improve OCR accuracy
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # OCR with whitelist (only letters and digits)
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    # Print recognized text
    print(f"Text from {os.path.basename(image_path)}: {text.strip()}")

    # Optional: display the image with OCR result
    display_img = img.copy()
    cv2.putText(display_img, text.strip(), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("OCR Result", display_img)
    cv2.waitKey(0)  # Press any key to move to next image

cv2.destroyAllWindows()
