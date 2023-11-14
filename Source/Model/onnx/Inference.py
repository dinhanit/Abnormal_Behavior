
from utils import *
from MediapipeFaceMesh import get_landmark_from_image

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
    results = softmax(results)
    if results[0][1]>=0.5:
        return 1
    else:
        return 0