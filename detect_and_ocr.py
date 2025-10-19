from ultralytics import YOLO
import cv2
import pytesseract

# Path to your tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the pretrained YOLO model
model = YOLO("yolov8n.pt")

# Path to your test image (you can use one from your train folder)
image_path = r"data\images\denoised\train\helmet5.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    # Run YOLO detection
    results = model(image)

    # Loop through detected objects
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Draw box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Class {cls} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Crop detected region (simulate number plate area)
            cropped = image[y1:y2, x1:x2]
            if cropped.size > 0:
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray)
                print(f"OCR Result for Class {cls}: {text.strip()}")

    # Show the final image
    cv2.imshow("YOLO + OCR Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
