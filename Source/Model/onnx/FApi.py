import numpy as np
from fastapi import FastAPI, UploadFile
import cv2
from Utils import *
from MediapipeFaceMesh import get_landmark_from_image
import numpy as np
app = FastAPI()

label =["Abnormal","Normal"]
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

def process_frame(frame_data):
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    label = Inference(frame)
    return label

@app.post("/process_frame/")
async def upload_frame(file: UploadFile):
    # if file.content_type and file.content_type.startswith("image/"):
        frame_data = await file.read()
        return process_frame(frame_data)

    
# app.mount("/", StaticFiles(directory="ui", html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)

