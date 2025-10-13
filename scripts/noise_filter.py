import cv2
import os

def denoise_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Simple noise reduction
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, denoised)
            print(f"✅ Processed: {filename}")

# Run for both folders
denoise_images(r"C:\Users\dell\OneDrive\Desktop\HelmetNumberplate\data\images\train",
               r"C:\Users\dell\OneDrive\Desktop\HelmetNumberplate\data\images\denoised\train")

denoise_images(r"C:\Users\dell\OneDrive\Desktop\HelmetNumberplate\data\images\val",
               r"C:\Users\dell\OneDrive\Desktop\HelmetNumberplate\data\images\denoised\val")

