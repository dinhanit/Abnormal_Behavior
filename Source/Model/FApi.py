"""
Create API
"""

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile
import torch
from Inference import Inference
from Param import DEVICE


app = FastAPI()


# Load the pre-trained model
model = torch.load('model/weight').to(DEVICE)
# If you want to load the model on the CPU, use the following line instead:
# model = torch.load('model/weight',map_location='cpu')

def process_frame(frame_data):
    """
    Process an uploaded frame and perform inference using a pre-trained model.

    Args:
        frame_data: Raw frame data as bytes.

    Returns:
        label: Inference result, typically a classification label or prediction.

    This function decodes the frame data, sends it to the pre-trained model, and returns the
    result of the inference.

    Example:
    ```
    frame_data = ...  # Load frame data from an image file or stream
    label = process_frame(frame_data)
    ```

    Note:
    Make sure the 'model' variable is properly loaded with the pre-trained model.

    """
    nparr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    label = Inference(model,frame)
    return label

@app.post("/process_frame/")
async def upload_frame(file: UploadFile):
    """
    Endpoint for uploading and processing image frames.

    Args:
        file: An uploaded image frame file.

    Returns:
        JSONResponse: A JSON response containing the result of the frame processing.

    This endpoint allows you to upload an image frame, which is then processed using the
    'process_frame' function. The result is returned as a JSON response.

    Example:
    Use a tool like cURL or a web application to send a POST request with an image file
    to this endpoint for processing.
    """
    frame_data = await file.read()
    return process_frame(frame_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
