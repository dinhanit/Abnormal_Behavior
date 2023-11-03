import mediapipe as mp
import math
import numpy as np
import cv2
class FaceMesh:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

        # Define the indices of the desired keypoints (0 to 467)
        self.keypoints = {
            "cheeks": [454, 234, 151, 152, 10, 376, 352, 433, 123, 147, 213, 58, 132, 288, 361],
            "right_iris": [469, 470, 471, 472],
            "left_iris": [474, 475, 476, 477]
        }

    def euclid_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def calculate_distances(self, points):
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = self.euclid_distance(points[i], points[j])
                distances.append(distance)
        return distances

    def detect_keypoints(self, image):
        ih, iw, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = self.face_mesh.process(rgb_image)

        if landmarks.multi_face_landmarks:
            distances = []
            for face_landmarks in landmarks.multi_face_landmarks:
                cheek_points = [
                    (int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih))
                    for i in self.keypoints["cheeks"]
                ]

                left_iris_landmarks = [face_landmarks.landmark[i] for i in self.keypoints["left_iris"]
                                      ]
                left_iris_center = (
                    int(np.mean([landmark.x * iw for landmark in left_iris_landmarks])),
                    int(np.mean([landmark.y * ih for landmark in left_iris_landmarks])),
                )

                right_iris_landmarks = [face_landmarks.landmark[i] for i in self.keypoints["right_iris"]
                                       ]
                right_iris_center = (
                    int(np.mean([landmark.x * iw for landmark in right_iris_landmarks])),
                    int(np.mean([landmark.y * ih for landmark in right_iris_landmarks])),
                )

                iris_center = (
                    (left_iris_center[0] + right_iris_center[0]) / 2,
                    (left_iris_center[1] + right_iris_center[1]) / 2
                )

                points = cheek_points + [left_iris_center, right_iris_center]
                distances.extend(self.calculate_distances(points))

            return np.array(distances, dtype=np.float32)
        else:
            return np.array([0.0] * 136)