import cv2
import numpy as np

def get_leaf_severity(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([5, 40, 40])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    diseased_pixels = np.sum(mask > 0)
    total_pixels = image.shape[0] * image.shape[1]

    severity = (diseased_pixels / total_pixels) * 100
    return severity, mask
