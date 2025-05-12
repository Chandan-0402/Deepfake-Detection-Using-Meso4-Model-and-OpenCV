import cv2
import numpy as np
import os
from mesonet_model import Meso4

# Load the pre-trained model
model = Meso4()
model.load_weights("weights/Meso4_DF.h5")

# Specify the image path
img_path = "C:/Users/Aayush/Desktop/deepfake_detector/test_image.jpg"  # Change this to the actual path if needed

# Read and validate the image
image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"Image not found or cannot be loaded: {img_path}")

# Preprocess the image
image = cv2.resize(image, (256, 256))
image = image.astype(np.float32) / 255.0
image = np.expand_dims(image, axis=0)

# Run prediction
prediction = model.predict(image)[0][0]
label = "FAKE" if prediction > 0.5 else "REAL"
confidence = prediction if label == "FAKE" else 1 - prediction

# Output the result
print("\n========== Deepfake Detection ==========")
print(f"[RESULT]     This image is likely: {label}")
print(f"[CONFIDENCE] Score: {confidence * 100:.2f}%")
