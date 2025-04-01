import cv2
import numpy as np
import os

# Global variables
bg = None
img_num = 0

# Dataset paths
BASE_PATH = r"D:\GitHub\Google-Meet-Sign-Language-Convertor\backend"
TRAIN_PATH = os.path.join(BASE_PATH, "Split Dataset", "train", "1")
VAL_PATH = os.path.join(BASE_PATH, "Split Dataset", "val", "1")
TEST_PATH = os.path.join(BASE_PATH, "Split Dataset", "test", "1")

# Ensure directories exist
os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(VAL_PATH, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None
    else:
        return thresholded, max(cnts, key=cv2.contourArea)

if __name__ == "__main__":
    aWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0

    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            break
        
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 100:
            print(f"Finding average, frame: {num_frames}")
            run_avg(gray, aWeight)
        else:
            hand = segment(gray)
            if hand is not None:
                thresholded, segmented = hand
                masked = cv2.bitwise_and(roi, roi, mask=thresholded)
                cv2.drawContours(clone, [segmented + (right, top)], -1, (255, 0, 255))
                cv2.imshow("Thresholded", thresholded)
                cv2.imshow("Masked", masked)
                
                save_path = ""
                if img_num <= 250:
                    save_path = os.path.join(TRAIN_PATH, f"{img_num+3000}.jpg")
                elif img_num <= 300:
                    save_path = os.path.join(VAL_PATH, f"{img_num+3000}.jpg")
                elif img_num <= 350:
                    save_path = os.path.join(TEST_PATH, f"{img_num+3000}.jpg")
                else:
                    break
                
                cv2.imwrite(save_path, masked)
                img_num += 1
                masked = cv2.resize(masked, (150, 150))
                masked = np.reshape(masked, (1, masked.shape[0], masked.shape[1], 3))
        
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
    
    camera.release()
    cv2.destroyAllWindows()