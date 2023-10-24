# import cv2
# import mediapipe as mp
# import numpy as np

# # Initialize MediaPipe Face Detection and Facial Landmarks models
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# # Define the indices of the desired keypoints (0 to 467)
# cheeks = [151, 152, 454, 234]
# right_iris = [469, 470, 471, 472]
# left_iris = [474, 475, 476, 477]

# desired_keypoint_indices = []
# desired_keypoint_indices.extend(cheeks)
# desired_keypoint_indices.extend(right_iris)
# desired_keypoint_indices.extend(left_iris)

# max_keypoints = len(desired_keypoint_indices)

# def get_landmark_from_image(image_path):
#     '''This function input an image and return landmarks in the image'''
#     # Load an image
#     image = cv2.imread(image_path)

#     # Convert the frame to RGB
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     landmarks = face_mesh.process(rgb_image)

#     if landmarks.multi_face_landmarks:
#         # Create a list to store data for this frame
#         frame_data = []
#         for face_landmarks in landmarks.multi_face_landmarks:
#             # Store index and landmark values for the desired keypoints
#             for index in desired_keypoint_indices:
#                 landmark = face_landmarks.landmark[index]
#                 frame_data.extend([landmark.x, landmark.y])

#             # Calculate the center of the left iris
#             left_iris_landmarks = [face_landmarks.landmark[i] for i in left_iris]
#             left_iris_center_x = np.mean([landmark.x for landmark in left_iris_landmarks])
#             left_iris_center_y = np.mean([landmark.y for landmark in left_iris_landmarks])
            
#             # Calculate the center of the right iris
#             right_iris_landmarks = [face_landmarks.landmark[i] for i in right_iris]
#             right_iris_center_x = np.mean([landmark.x for landmark in right_iris_landmarks])
#             right_iris_center_y = np.mean([landmark.y for landmark in right_iris_landmarks])
            

#         # Store only the cheek and iris center points
#         frame_data = frame_data[:8]
#         frame_data.extend([left_iris_center_x, left_iris_center_y])
#         frame_data.extend([right_iris_center_x, right_iris_center_y])
#         return np.array(frame_data, dtype=np.float32)
#     else:
#         return np.array([0.0, 0.0] * 6)  # 6 points: 4 cheeks and 2 iris centers

# landmark_data = get_landmark_from_image('image_10.jpg')
# print(landmark_data)


'''
TEST NEW METHOD WITH DISTANCE (ADD IRIS)
'''
# import cv2
# import mediapipe as mp
# import numpy as np
# import math
# # Initialize MediaPipe Face Detection and Facial Landmarks models
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# # Define the indices of the desired keypoints (0 to 467)
# cheeks = [454, 234, 151, 152, 10, 376, 352, 433, 123, 147, 213, 58, 132, 288, 361]#
# right_iris = [469, 470, 471, 472]
# left_iris = [474, 475, 476, 477]

# desired_keypoint_indices = []
# desired_keypoint_indices.extend(cheeks)
# desired_keypoint_indices.extend(right_iris)
# desired_keypoint_indices.extend(left_iris)

# max_keypoints = len(desired_keypoint_indices)

# def euclid_distance(point1, point2):
#     x1, y1 = point1
#     x2, y2 = point2
#     distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#     return distance

# def calculate_distances(points):
#     distances = []
#     for i in range(len(points)):
#         for j in range(i + 1, len(points)):
#             distance = euclid_distance(points[i], points[j])
#             distances.append(distance)
#     return distances

# def get_landmark_from_image(image_path):
#     # Load an image
#     image = cv2.imread(image_path)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     landmarks = face_mesh.process(rgb_image)

#     if landmarks.multi_face_landmarks:
#         #frame_data = []
#         ih, iw, _ = image.shape 
#         for face_landmarks in landmarks.multi_face_landmarks:
#             cheek_points = [(face_landmarks.landmark[i].x *iw,
#                             face_landmarks.landmark[i].y *ih) for i in cheeks]

#             left_iris_landmarks = [face_landmarks.landmark[i] for i in left_iris]
#             left_iris_center = (
#                 np.mean([landmark.x *iw for landmark in left_iris_landmarks]),
#                 np.mean([landmark.y *ih for landmark in left_iris_landmarks]),
#             )

#             right_iris_landmarks = [face_landmarks.landmark[i] for i in right_iris]
#             right_iris_center = (
#                 np.mean([landmark.x *iw for landmark in right_iris_landmarks]),
#                 np.mean([landmark.y *ih for landmark in right_iris_landmarks]),
#             )

#             iris_center = (
#                 (left_iris_center[0] + right_iris_center[0]) / 2,
#                 (left_iris_center[1] + right_iris_center[1]) / 2
#             )

#             points = cheek_points + [left_iris_center, right_iris_center]
#             #print(points)
#             distances = calculate_distances(points)

#         return np.array(distances, dtype=np.float32) 
#     else:
#         return np.array([0.0] * 136)

# image_path = 'image_1.jpg'  # Replace with your image path
# distances = get_landmark_from_image(image_path)

# print("Distances between points:")
# print(distances)



import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Face Detection and Facial Landmarks models
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Define the indices of the desired keypoints (0 to 467)
cheeks = [454, 234, 151, 152, 10, 376, 352, 433, 123, 147, 213, 58, 132, 288, 361]
right_iris = [469, 470, 471, 472]
left_iris = [474, 475, 476, 477]

desired_keypoint_indices = []
desired_keypoint_indices.extend(cheeks)
desired_keypoint_indices.extend(right_iris)
desired_keypoint_indices.extend(left_iris)

max_keypoints = len(desired_keypoint_indices)

def euclid_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def calculate_distances(points):
    # Use NumPy to calculate pairwise distances
    return np.linalg.norm(points[:, np.newaxis] - points, axis=2)

def get_landmark_from_image(image_path):
    # Load an image
    image = cv2.imread(image_path)
    ih, iw, _ = image.shape

    # Convert the frame to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    landmarks = face_mesh.process(rgb_image)

    if landmarks.multi_face_landmarks:
        distances = []
        for face_landmarks in landmarks.multi_face_landmarks:
            landmarks_2d = np.array([(lm.x * iw, lm.y * ih) for i, lm in enumerate(face_landmarks.landmark)])

            cheek_points = landmarks_2d[cheeks]
            left_iris_landmarks = landmarks_2d[left_iris]
            right_iris_landmarks = landmarks_2d[right_iris]

            iris_center = np.mean([left_iris_landmarks, right_iris_landmarks], axis=0)

            points = np.vstack((cheek_points, iris_center))
            distance_matrix = calculate_distances(points)

            # Flatten the upper triangular part of the distance matrix
            distances.extend(distance_matrix[np.triu_indices(len(points), k=1)])

        return np.array(distances, dtype=np.float32)
    else:
        return np.array([0.0] * 136)

image_path = 'image_31.jpg'  # Replace with your image path
distances = get_landmark_from_image(image_path)

print("Distances between points:")
print(len(distances))
