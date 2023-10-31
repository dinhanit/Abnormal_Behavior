import cv2
import mediapipe as mp
import numpy as np
'''JUST RETURN THE LANDMARK POINT (WITH 4 LANDMARK IRIS)'''
# Initialize MediaPipe Face Detection and Facial Landmarks models
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks = True)

# Define the indices of the desired keypoints (0 to 467)
cheeks = [454, 234, 151, 152, 10, 376, 352, 433, 123, 147, 213, 58, 132, 288, 361]
left_iris = [474, 475, 476, 477]
right_iris = [469, 470, 471, 472]
desired_keypoint_indices = []  # Modify this list to choose your desired keypoints
desired_keypoint_indices.extend(cheeks)
desired_keypoint_indices.extend(left_iris)
desired_keypoint_indices.extend(right_iris)
max_keypoints = len(desired_keypoint_indices) #############

def get_landmark_from_image(image_path):
    '''This function input an image and return landmarks in the image'''
    # Load an image
    image = cv2.imread(image_path)

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
        return np.array([0.0, 0.0, 0.0] * max_keypoints) ###############
    
    
    