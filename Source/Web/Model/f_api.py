import numpy as np

from Inference import Inference
import torch
from param import *
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
app = FastAPI()

model = torch.load('model/weight').to(DEVICE)

def process_frame(frame_data):
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    label = Inference(model,frame)
    print(label)
    return 0

@app.post("/process_frame/")
async def upload_frame(file: UploadFile):
    if file.content_type and file.content_type.startswith("image/"):
        frame_data = await file.read()
        process_frame(frame_data)
        return JSONResponse(content={"message": "Frame received and processed successfully"})
    else:
        raise HTTPException(status_code=400, detail="Invalid file type, only images are supported")
    
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.2", port=8000)