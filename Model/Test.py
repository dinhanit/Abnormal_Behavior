from BaseModel import BinaryClassifier
from keyPointDetect import DetectKeyPoint
import torch
import torch.nn.functional as F
import cv2

model = torch.load(".model/Weight")

def Inferent(img):
    global model
    label = ['Abnormal','Normal']
    kp = DetectKeyPoint(img)
    kp = kp.reshape(1,-1)
    model.eval()
    with torch.no_grad():
        output = model(torch.Tensor(kp))
    probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return label[predicted_class]

video = "head2.mp4"
cap = cv2.VideoCapture(video)
desired_fps = 1
cap.set(cv2.CAP_PROP_FPS, desired_fps)
while True:
    ret, frame = cap.read()
    label = Inferent(frame)
    
    text = label
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    font_scale = 1

    thickness = 2
    if label=='Normal':
        font_color = (0, 255, 0)
    else:
        font_color = (0,0,255)
    frame = cv2.putText(frame, text, org, font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.imshow('Camera Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
