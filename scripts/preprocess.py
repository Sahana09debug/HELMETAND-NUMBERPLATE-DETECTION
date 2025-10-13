import cv2
import os

def preprocess_images(input_folder, output_folder, size=(640, 640)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Skipping (not readable): {filename}")
                continue

            resized = cv2.resize(image, size)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")
            cv2.imwrite(output_path, resized)
            print(f"Processed: {filename}")

input_dir = r"C:\Users\dell\OneDrive\Desktop\HelmetNumberplate\data\images\train"
output_dir = r"C:\Users\dell\OneDrive\Desktop\HelmetNumberplate\dataset\train"
preprocess_images(input_dir, output_dir)
