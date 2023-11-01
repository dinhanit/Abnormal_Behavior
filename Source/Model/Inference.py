import torch
import torch.nn.functional as F
from Param import DEVICE
from MediapipeFaceMesh import get_landmark_from_image

def Inference(model,img,threshold=0.5):
    """
    Perform inference using a deep learning model on facial landmarks.

    Args:
        model (torch.nn.Module): A pre-trained deep learning model for inference.
        img (np.ndarray): Input image containing facial landmarks.

    Returns:
        str: A classification label indicating the result of the inference.

    This function takes a pre-trained deep learning model, an input image containing facial landmarks,
    and performs inference to classify the input as either 'Abnormal' or 'Normal'. The model should
    be designed for binary classification.

    Example:
    ```
    from your_module import Inference

    model = ...  # Load your pre-trained model
    image = ...  # Load an image containing facial landmarks
    result = Inference(model, image)
    ```

    Note:
    - The 'model' should be properly initialized and loaded with your pre-trained model.
    - The 'img' argument should contain facial landmarks, and the model should be trained
      to make binary classification predictions.
    - If the model is not confident in its prediction, it may return 'Abnormal'.
    """
    label = ['Abnormal','Normal']
    kp = get_landmark_from_image(img)
    try:
        kp = kp.reshape(1,-1)
        model.eval()
        with torch.no_grad():
            output = model(torch.Tensor(kp).to(DEVICE))
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        if probabilities[0][1]>=threshold:
            predicted_class = 1
        else:
            predicted_class = 0
        return label[predicted_class]
    except:
        return label[0]