import sys
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = "pneumonia_cnn_model.h5"
IMAGE_SIZE = (150, 150)

model = load_model(MODEL_PATH)

def predict_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        if prediction[0][0] > 0.5:
            return "Pneumonia Detected"
        else:
            return "Normal"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        image_path = sys.argv[1]
        result = predict_image(image_path)
        print("🩺 Prediction:", result)
