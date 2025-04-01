import cv2
import tensorflow as tf
import numpy as np
import pyttsx3 as p
import os

# Global variables
bg = None
engine = p.init()

# Set correct paths
BASE_PATH = r"D:\GitHub\Google-Meet-Sign-Language-Convertor\backend"
MODEL_PATH = os.path.join(BASE_PATH, "best_model.h5")
CLASSES_PATH = os.path.join(BASE_PATH, "classes.npy")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels safely
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
labels = np.load(CLASSES_PATH).item()
np.load = np_load_old
labels = {v: k for k, v in labels.items()}  # Reverse dictionary

def run_avg(image, aWeight):
    """Compute running average of the background."""
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    """Segment the hand from the background."""
    global bg
    if bg is None:
        return None
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return (thresholded, max(cnts, key=cv2.contourArea)) if cnts else None

if __name__ == "__main__":
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0

    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            print("Error: Camera not detected!")
            break

        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 100:
            print(f"Calibrating background... Frame: {num_frames}")
            run_avg(gray, aWeight)
        else:
            hand = segment(gray)
            if hand:
                thresholded, segmented = hand
                masked = cv2.bitwise_and(roi, roi, mask=thresholded)

                # Draw contour
                cv2.drawContours(clone, [segmented + (right, top)], -1, (255, 0, 255), 2)
                cv2.imshow("Thresholded", thresholded)

                # Resize & prepare input
                masked = cv2.resize(masked, (150, 150))
                masked = np.reshape(masked, (1, 150, 150, 3))  # Keep 3 channels

                # Predict sign language gesture
                pred = model.predict(masked)
                predicted_class_index = np.argmax(pred, axis=1)[0]

                # Display & speak prediction
                if predicted_class_index in labels:
                    prediction_text = labels[predicted_class_index]
                    cv2.putText(clone, prediction_text, (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print(f"Predicted Sign: {prediction_text}")
                    engine.say(prediction_text)
                    engine.runAndWait()
                else:
                    print("Warning: Predicted class not found in labels!")

        # Draw ROI box
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frames += 1

        # Display feed
        cv2.imshow("Video Feed", clone)
        
        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
