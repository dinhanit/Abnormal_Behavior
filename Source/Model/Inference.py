import torch
import torch.nn.functional as F
import cv2
from param import DEVICE
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Detection and Facial Landmarks models
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks = True)

# Define the indices of the desired keypoints (0 to 467)
cheeks = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
left_eyes = [33, 159, 133, 145,]
right_eyes = [263, 386, 374, 362]
nose = [1, 2, 5, 3, 248]
left_iris = [474, 475, 476, 477]
right_iris = [469, 470, 471, 472]

desired_keypoint_indices = []  # Modify this list to choose your desired keypoints
desired_keypoint_indices.extend(cheeks)
desired_keypoint_indices.extend(nose)
desired_keypoint_indices.extend(left_iris)
desired_keypoint_indices.extend(right_iris)
desired_keypoint_indices.extend(left_eyes)
desired_keypoint_indices.extend(right_eyes)
max_keypoints = len(desired_keypoint_indices)

def DetectKeyPoint(image):
    # Convert the frame to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    landmarks = face_mesh.process(rgb_image)

    if landmarks.multi_face_landmarks:
        # Create a list to store data for this frame
        frame_data = []
        for face_landmarks in landmarks.multi_face_landmarks:
            # Store index and landmark values for the desired keypoints
            for index in desired_keypoint_indices:
                landmark = face_landmarks.landmark[index]
                frame_data.extend([landmark.x, landmark.y, landmark.z])

        return np.array(frame_data, dtype=np.float32)
    else:
        return np.array([0.0, 0.0, 0.0] * max_keypoints)
    

def Inference(model,img):
    label = ['Abnormal','Normal']
    kp = DetectKeyPoint(img)
    kp = kp.reshape(1,-1)
    model.eval()
    with torch.no_grad():
        output = model(torch.Tensor(kp).to(DEVICE))
    probabilities = F.softmax(output, dim=1)
    # predicted_class = torch.argmax(probabilities, dim=1).item()
    print(probabilities)
    if probabilities[0][1]>=0.35:
        predicted_class = 1
    else:
        predicted_class = 0
    return label[predicted_class]