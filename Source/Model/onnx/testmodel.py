import numpy as np
import cv2
import numpy as np
from Inference import Inference
import cv2

cap = cv2.VideoCapture("video_test2.mp4") 
# cap = cv2.VideoCapture(0) 

position = (50, 50)  # (x, y) coordinates
font = 0
font_scale = 1
font_thickness = 2

label =["Abnormal","Normal"]
color =[(0,0,255),(0,255,0)]
while True:
    ret, frame = cap.read()
    if not ret:
        break  
    idx = Inference(frame)
    cv2.putText(frame, label[idx], position, font, font_scale, color[idx], font_thickness)
    cv2.imshow("Camera Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
