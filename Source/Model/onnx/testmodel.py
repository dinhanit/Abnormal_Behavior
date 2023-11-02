import numpy as np
from Utils import *
from MediapipeFaceMesh import get_landmark_from_image
import cv2
import numpy as np

session = load_session("model.onnx")

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def Inference(img):
    cov_img = get_landmark_from_image(img)
    if np.all(cov_img==0):
        return 0
    
    input_data = np.array([[cov_img]])
    results = infer(session, input_data)
    return np.argmax(softmax(results))

import cv2
position = (50, 50)  # (x, y) coordinates
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

# Use cv2.putText to add the text to the image
cap = cv2.VideoCapture("video_test2.mp4") 
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
