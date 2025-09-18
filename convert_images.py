from PIL import Image
import os

folder = r"C:\Users\dell\OneDrive\Desktop\HelmetNumberplate\data\images\train"

for file in os.listdir(folder):
    if file.endswith(".png"):
        img_path = os.path.join(folder, file)
        try:
            img = Image.open(img_path).convert("RGB")
            new_path = os.path.join(folder, file.replace(".png", ".jpg"))
            img.save(new_path, "JPEG")
            print(f"✅ Converted {file} -> {new_path}")
        except Exception as e:
            print(f"❌ Error with {file}: {e}")
