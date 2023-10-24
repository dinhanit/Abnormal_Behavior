from Inference import Inference
import cv2
import torch
from param import DEVICE

model = torch.load(".model/Weight").to(DEVICE)

font_color = {"Normal" : (0, 255, 0),
              "Abnormal": (0,0,255)}

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
font_scale = 1
thickness = 2

op = "head1.mp4"
cap = cv2.VideoCapture(op)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()


while True:
    ret, frame = cap.read()
    label = Inference(model,frame)
    frame = cv2.putText(frame, label, org, font, font_scale, font_color[label], thickness, cv2.LINE_AA)
    cv2.imshow('Camera Stream', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
