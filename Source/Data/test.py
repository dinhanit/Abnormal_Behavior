import numpy as np
from Pre_Image import get_landmark_from_image
import cv2
def distance(keypoint,keypoints):
    n = len(keypoints)
    np_1 = np.array([keypoint]*n)
    np_2 = np.array(keypoints)
    return ((np_1[:,0]-np_2[:,0])**2 + (np_1[:,1]-np_2[:,1])**2)**(1/2)

def calculate_distance(keypoints):
    return np.array([distance(i,keypoints) for i in keypoints])

def Diff(img1,img2):
    kp1 = get_landmark_from_image(img1)
    kp2 = get_landmark_from_image(img2)
    d_kp1 = calculate_distance(kp1)
    d_kp2 = calculate_distance(kp2)
    return d_kp1-d_kp2

img_normal = cv2.imread("image_10.jpg")
img_abnormal = cv2.imread("image_31.jpg")
print("Differ")
print(Diff(img_normal,img_abnormal)[0].shape)





