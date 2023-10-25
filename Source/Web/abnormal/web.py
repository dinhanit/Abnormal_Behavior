from .BaseModel import BinaryClassifier
from .keyPointDetect import DetectKeyPoint
import torch
import torch.nn.functional as F
import cv2
from .param import DEVICE

model = torch.load("abnormal/Model/.model/Weight").to(DEVICE)
def Inference(img):
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
    if probabilities[0][1]<0.001:
        predicted_class = 0
    else:
        predicted_class = 1
    return label[predicted_class]