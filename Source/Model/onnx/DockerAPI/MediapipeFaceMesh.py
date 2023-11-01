import cv2
import mediapipe as mp
import numpy as np
import math
'''
TEST NEW METHOD WITH DISTANCE (ADD IRIS)
'''
# Initialize MediaPipe Face Detection and Facial Landmarks models
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Define the indices of the desired keypoints (0 to 467)
#cheeks = [454, 234, 151, 152, 10, 376, 352, 433, 123, 147, 213, 58, 132, 288, 361]#
right_iris = [469, 470, 471, 472]
left_iris = [474, 475, 476, 477]
##new points
cheeks = [10, 32, 140, 148, 152, 171, 175, 176, 199, 200, 201, 208, 332, 338, 396, 421, 428] 
desired_keypoint_indices = []
desired_keypoint_indices.extend(cheeks)
desired_keypoint_indices.extend(right_iris)
desired_keypoint_indices.extend(left_iris)
max_keypoints = len(desired_keypoint_indices)


def resize_image(img, width = 640, height = 480):

    h, w = img.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = w / h

    if aspect_ratio > width / height:
        # If the image is wider than the target aspect ratio, crop the sides
        new_width = int(height * aspect_ratio)
        new_height = height
    else:
        # If the image is taller than the target aspect ratio, crop the top and bottom
        new_width = width
        new_height = int(width / aspect_ratio)

    # Resize the image to the new dimensions
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Calculate the cropping offsets
    x_offset = (new_width - width) // 2
    y_offset = (new_height - height) // 2

    # Crop the image to the target dimensions
    cropped = resized[y_offset:y_offset + height, x_offset:x_offset + width]

    return cropped

def euclid_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def calculate_distances(points):
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = euclid_distance(points[i], points[j])
            distances.append(distance)
    return distances

def get_landmark_from_image(image):
    image = resize_image(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = face_mesh.process(rgb_image)

    if landmarks.multi_face_landmarks:
        #frame_data = []
        ih, iw, _ = image.shape 
        for face_landmarks in landmarks.multi_face_landmarks:
            cheek_points = [(face_landmarks.landmark[i].x *iw,
                            face_landmarks.landmark[i].y *ih) for i in cheeks]

            left_iris_landmarks = [face_landmarks.landmark[i] for i in left_iris]
            left_iris_center = (
                np.mean([landmark.x *iw for landmark in left_iris_landmarks]),
                np.mean([landmark.y *ih for landmark in left_iris_landmarks]),
            )

            right_iris_landmarks = [face_landmarks.landmark[i] for i in right_iris]
            right_iris_center = (
                np.mean([landmark.x *iw for landmark in right_iris_landmarks]),
                np.mean([landmark.y *ih for landmark in right_iris_landmarks]),
            )

            iris_center = (
                (left_iris_center[0] + right_iris_center[0]) / 2,
                (left_iris_center[1] + right_iris_center[1]) / 2
            )

            points = cheek_points + [left_iris_center, right_iris_center]
            #print(points)
            distances = calculate_distances(points)

        return np.array(distances, dtype=np.float32) 
    else:
        return np.array([0.0] * 171)

