"""
To check actual performance by label directly
"""

import torch
import cv2
from Param import DEVICE
from Inference import Inference

MODEL = torch.load("model/weight").to(DEVICE)

FONT_COLOR = {"Normal": (0, 255, 0), "Abnormal": (0, 0, 255)}

FONT = cv2.FONT_HERSHEY_SIMPLEX
ORG = (50, 50)
FONT_SCALE = 1
THICKNESS = 2

VIDEO_FILE = 'video_test2.mp4'
cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()


FRAME_COUNT = 0
TRUE_NEGATIVES = 0
TRUE_POSITIVES = 0
FALSE_POSITIVES = 0
FALSE_NEGATIVES = 0
TRUE_PREDICTIONS = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    FRAME_COUNT += 1
    label = Inference(MODEL, frame)
    frame = cv2.putText(frame, label, ORG, FONT, FONT_SCALE, FONT_COLOR[label], THICKNESS, cv2.LINE_AA)
    cv2.imshow('Video Stream', frame)
    key = cv2.waitKey(0)

    if key == ord('1') and label == "Abnormal":
        TRUE_POSITIVES += 1
        TRUE_PREDICTIONS +=1
    elif key == ord('0') and label == "Normal":
        TRUE_NEGATIVES
        TRUE_PREDICTIONS +=1
    elif label == "Abnormal":
        FALSE_POSITIVES += 1
    elif label == "Normal":
        FALSE_NEGATIVES += 1

    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Total Frames Processed:", FRAME_COUNT)
print("True Predictions:", TRUE_PREDICTIONS)

PRECISION = TRUE_POSITIVES / (TRUE_POSITIVES + FALSE_POSITIVES)
RECALL = TRUE_POSITIVES / (TRUE_POSITIVES + FALSE_NEGATIVES)
F1_SCORE = 2 * (PRECISION * RECALL) / (PRECISION + RECALL)
ACCURACY = (TRUE_POSITIVES + TRUE_NEGATIVES) / (TRUE_POSITIVES + TRUE_NEGATIVES + FALSE_POSITIVES + FALSE_NEGATIVES)

print("Precision: {:.2%}".format(PRECISION))
print("Recall: {:.2%}".format(RECALL))
print("F1 Score: {:.2%}".format(F1_SCORE))
print("Accuracy: {:.2%}".format(ACCURACY))
