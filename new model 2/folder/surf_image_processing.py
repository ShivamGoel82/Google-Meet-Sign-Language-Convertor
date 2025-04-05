import numpy as np
import cv2
from matplotlib import pyplot as plt

def func(path):    
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (128, 128))
    converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerBoundary = np.array([0, 40, 30], dtype="uint8")
    upperBoundary = np.array([43, 255, 254], dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask, 0.5, skinMask, 0.5, 0.0)
    skinMask = cv2.medianBlur(skinMask, 5)

    skin = cv2.bitwise_and(converted2, converted2, mask=skinMask)
    img2 = cv2.Canny(skin, 60, 60)
    img2 = cv2.resize(img2, (256, 256))

    # ✅ Using ORB instead of SURF
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img2, None)

    img2 = cv2.drawKeypoints(img2, kp, None, (0, 255, 0), 4)
    # plt.imshow(img2), plt.show()

    print(len(des) if des is not None else 0)
    return des

def func2(path):    
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (128, 128))
    converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerBoundary = np.array([0, 40, 30], dtype="uint8")
    upperBoundary = np.array([43, 255, 254], dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask, 0.5, skinMask, 0.5, 0.0)
    skinMask = cv2.medianBlur(skinMask, 5)

    skin = cv2.bitwise_and(converted2, converted2, mask=skinMask)
    img2 = cv2.Canny(skin, 60, 60)
    img2 = cv2.resize(img2, (256, 256))

    # ✅ Correct ORB usage
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img2, None)

    img2 = cv2.drawKeypoints(img2, kp, None, color=(0, 255, 0), flags=0)
    # plt.imshow(img2), plt.show()

    return des

# Example usage
# des = func("001.jpg")
