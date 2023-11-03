from Inference import Inference
import cv2
import torch
from param import DEVICE
import argparse
parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--source', type=str, default='head1.mp4', help='Path to save the trained model')
args = parser.parse_args()

model = torch.load("abmodel/Weight").to(DEVICE)

font_color = {"Normal" : (0, 255, 0),
              "Abnormal": (0,0,255)}

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
font_scale = 1
thickness = 2

op = args.source
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()


while True:
    ret, frame = cap.read()
    label = Inference(model,frame)
    frame = cv2.putText(frame, label, org, font, font_scale, font_color[label], thickness, cv2.LINE_AA)
    cv2.imshow('Camera Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
