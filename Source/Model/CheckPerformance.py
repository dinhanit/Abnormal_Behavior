import torch
import torch.nn.functional as F
import cv2
from Param import DEVICE
from Inference import Inference

model = torch.load("model/Weight").to(DEVICE)

font_color = {"Normal": (0, 255, 0), "Abnormal": (0, 0, 255)}

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
font_scale = 1
thickness = 2

video_file = 'video_test2.mp4'
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

true_predicted = 0
true_normal = 0
predicted_normal = 0
frame_count = 0
false_positives = 0
false_negatives = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    label = Inference(model, frame)
    frame = cv2.putText(frame, label, org, font, font_scale, font_color[label], thickness, cv2.LINE_AA)
    cv2.imshow('Video Stream', frame)
    key = cv2.waitKey(0)

    if key == ord('1') and label == "Abnormal":
        true_predicted += 1
    elif key == ord('0') and label == "Normal":
        true_predicted += 1
        true_normal += 1
    elif label == "Abnormal":
        false_negatives += 1
    elif label == "Normal":
        false_positives += 1

    if key & 0xFF == ord('q'):
        break
    else:
        pass

cap.release()
cv2.destroyAllWindows()

print("Total Frames Processed:", frame_count)
print("True Predictions:", true_predicted)

precision = true_predicted / (true_predicted + false_positives)
recall = true_predicted / (true_predicted + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)
true_negatives = frame_count - true_predicted - false_positives - false_negatives
accuracy = (true_predicted + true_negatives) / frame_count

print("Precision: {:.2%}".format(precision))
print("Recall: {:.2%}".format(recall))
print("F1 Score: {:.2%}".format(f1_score))
print("Accuracy: {:.2%}".format(accuracy))