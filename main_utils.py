import cv2
import numpy as np
from model_utils import *

def get_boundary_points(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), dtype=int)

    frame = cv2.dilate(frame, kernel, iterations=1)
    frame = cv2.erode(frame, kernel, iterations=1)
    frame = cv2.dilate(frame, kernel, iterations=2)
    frame= cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=5)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    boxes = []
    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if h >= frame.shape[0] * 0.05 and w >= frame.shape[1] * 0.05:
            for x1, y1, w1, h1 in boxes:
                if abs(x1 - x) <= max(w, w1) and abs(y - y1)  <= max(h, h1):
                    break
            else:
                boxes.append([x, y, w, h])

    return boxes, frame


def get_resized_images(boxes, frame, percent=0.1):
    
    images = []
    for x, y, w, h in boxes:

        w, h = max(w, h) + int(w * 2 * percent), max(w, h) + int(h * 2 * percent)
        x, y = max(0, int(x - (w * percent))), max(0, int(y - h * percent))

        img = np.expand_dims(cv2.resize(frame[y:y+h, x:x+w], (28, 28)), axis=-1)
        images.append(img)

    return np.array(images)


def draw_bounding_boxes(boxes, predictions, img):

    for (x, y, w, h), p in zip(boxes, predictions.keys()):

        cv2.rectangle(img, (x, y), (x + w, y + h), (101, 255, 111), 2)

        cv2.putText(img, str(p) + "@" + str(predictions[p]) + "%", (x + w, int(y + h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 101, 125), 1, cv2.LINE_AA)

    return img
