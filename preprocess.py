import os
from PIL import Image

IMAGE_SIZE = (150, 150)
RAW_DATA_PATH = os.path.abspath("dataset_raw")
PROCESSED_DATA_PATH = os.path.abspath("processed_images")

for split in ["train", "val", "test"]:
    for category in ["Normal", "Pneumonia"]:
        os.makedirs(os.path.join(PROCESSED_DATA_PATH, split, category), exist_ok=True)

def clean_and_resize_images(source, destination):
    for category in ["Normal", "Pneumonia"]:
        source_folder = os.path.join(source, category)
        dest_folder = os.path.join(destination, category)

        if not os.path.exists(source_folder):
            print(f"❌ Skipping missing folder: {source_folder}")
            continue

        for img_name in os.listdir(source_folder):
            src_path = os.path.join(source_folder, img_name)
            dst_path = os.path.join(dest_folder, img_name)

            try:
                with Image.open(src_path) as img:
                    img.verify()

                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(IMAGE_SIZE)
                    img.save(dst_path)

            except Exception as e:
                print(f"❌ Skipping corrupted image: {img_name} - {e}")

for split in ["train", "val", "test"]:
    clean_and_resize_images(os.path.join(RAW_DATA_PATH, split), os.path.join(PROCESSED_DATA_PATH, split))

print("✅ Data Cleaning & Resizing Complete!")
