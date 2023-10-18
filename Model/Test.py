from BaseModel import BinaryClassifier
from keyPointDetect import DetectKeyPoint
import torch
import torch.nn.functional as F
import cv2
from param import DEVICE

model = torch.load(".model/Weight")
 
def Inferent(img):
    global model
    label = ['Abnormal','Normal']
    kp = DetectKeyPoint(img)
    kp = kp.reshape(1,-1)
    model.eval()
    with torch.no_grad():
        output = model(torch.Tensor(kp).to(DEVICE))
    probabilities = F.softmax(output, dim=1)
    # predicted_class = torch.argmax(probabilities, dim=1).item()
    print(probabilities)
    if probabilities[0][0]>=0.8:
        predicted_class = 0
    else:
        predicted_class = 1
    return label[predicted_class]



font_color = {"Normal" : (0, 255, 0),
              "Abnormal": (0,0,255)}

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
font_scale = 1
thickness = 2

op = 0
cap = cv2.VideoCapture(op)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()


while True:
    ret, frame = cap.read()
    label = Inferent(frame)
    frame = cv2.putText(frame, label, org, font, font_scale, font_color[label], thickness, cv2.LINE_AA)
    cv2.imshow('Camera Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
