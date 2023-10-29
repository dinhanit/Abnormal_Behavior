import numpy as np
from test_Pre_Image import get_landmark_from_image 
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
result = Diff(img_normal,img_abnormal)
print(result)

# import pandas as pd 
# df = pd.DataFrame(result)
# df.to_csv('test.csv', index=False)

# import pandas as pd

# # Assuming 'result' is your DataFrame
# df = pd.DataFrame(result)

# # Set custom column names
# custom_column_names = list(range(0, 478))  # Replace with your actual column names
# df.columns = custom_column_names

# # Set custom row names
# custom_row_names = list(range(0, 478)) # Replace with your desired row names
# df.index = custom_row_names
# df.index.name = "Index"
# # Now, your DataFrame has custom row names and custom column names.

# # Save the DataFrame to a CSV file
# df.to_csv('test.csv', index=True, header=True)





