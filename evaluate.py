import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score

# Constants
DATASET_DIR = os.path.abspath("processed_images")
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
MODEL_PATH = "pneumonia_cnn_model.h5"
THRESHOLD = 0.6  # 👈 Adjust this for better precision/accuracy balance

# Load model
model = load_model(MODEL_PATH)


# Preprocess test images
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_data = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False  
)

# Evaluate loss and accuracy
loss, accuracy = model.evaluate(test_data)
print(f"✅ Test Loss: {loss:.4f}")
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# Predict with threshold tuning
y_pred_prob = model.predict(test_data)
y_pred = (y_pred_prob > THRESHOLD).astype(int).flatten()
y_true = test_data.classes

# Metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ Adjusted Accuracy: {acc:.4f}")

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_data.class_indices))
