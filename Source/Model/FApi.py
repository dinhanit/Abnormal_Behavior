import numpy as np
from Inference import Inference
import torch
from param import *
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
app = FastAPI()

# model = torch.load('model/weight').to(DEVICE)
model = torch.load('model/weight',map_location='cpu')

def process_frame(frame_data):
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    label = Inference(model,frame)
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