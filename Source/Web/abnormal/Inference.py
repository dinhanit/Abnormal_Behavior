import torch
import torch.nn.functional as F
from .param import DEVICE
from .MediapipeFaceMesh import FaceMesh

FM = FaceMesh()

def Inference(model,img):
    label = ['Abnormal','Normal']
    kp = FM.detect_keypoints(img)
    kp = kp.reshape(1,-1)
    model.eval()
    with torch.no_grad():
        output = model(torch.Tensor(kp).to(DEVICE))
    probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    print(probabilities)
    if probabilities[0][1]>=0.5:
        predicted_class = 1
    else:
        predicted_class = 0
    return label[predicted_class]