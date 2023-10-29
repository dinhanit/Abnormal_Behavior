import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection and Facial Landmarks models
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def get_landmark_from_image(image):
    '''This function inputs an image and returns all landmarks with their indices in the image'''
    # Convert the frame to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    landmarks = face_mesh.process(rgb_image)
    iw, ih, _ = image.shape

    if landmarks.multi_face_landmarks:
        for face_landmarks in landmarks.multi_face_landmarks:
            frame_data = []
            for i, landmark in enumerate(face_landmarks.landmark):
                x, y = landmark.x * iw, landmark.y * ih
                frame_data.append((x, y))

        return np.array(frame_data, dtype=np.float32)
    else:
        # Return an empty list if no faces are detected
        return np.array([])

# image = cv2.imread("image_10.jpg")
# print(get_landmark_from_image(image).shape)