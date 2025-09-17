from PIL import Image
import os

# Change this to your train folder path
input_dir = r"C:\Users\dell\HelmetPlateProject\data\images\train"

for file in os.listdir(input_dir):
    if file.endswith(".png"):
        try:
            img_path = os.path.join(input_dir, file)
            img = Image.open(img_path).convert("RGB")  # ensure 8-bit RGB
            img.save(img_path)  # overwrite with fixed version
            print(f"Fixed: {file}")
        except Exception as e:
            print(f"Error fixing {file}: {e}")
