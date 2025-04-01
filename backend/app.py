from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
import os

# Initialize Flask App
app = Flask(__name__)

# Enable CORS for all routes (Restrict if needed)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allows requests from any origin

# Load the trained model
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "best_model.h5")
CLASSES_PATH = os.path.join(BASE_PATH, "classes.npy")

model = tf.keras.models.load_model(MODEL_PATH)
labels = np.load(CLASSES_PATH, allow_pickle=True).item()
labels = {v: k for k, v in labels.items()}  # Reverse dictionary

# OpenCV Video Capture
camera = cv2.VideoCapture(0)

def get_prediction():
    ret, frame = camera.read()
    if not ret:
        return None

    # Process the frame (Convert to grayscale, resize, etc.)
    roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (150, 150))
    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    roi = np.reshape(roi, (1, 150, 150, 3))  # Reshape for model

    # Predict the sign
    pred = model.predict(roi)
    predicted_class_index = np.argmax(pred, axis=1)[0]

    return labels.get(predicted_class_index, "Unknown")

@app.route("/predict", methods=["GET"])
def predict():
    prediction = get_prediction()
    if prediction:
        return jsonify({"text": prediction})
    return jsonify({"text": ""})  # Empty response if no prediction

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
