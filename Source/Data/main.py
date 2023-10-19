import numpy as np

# Load the data from the .npz file
data = np.load('Data_train.npz')
print(data)
# Access the keypoint array
keypoint_array = data['labels']
print(keypoint_array)
# print(keypoint_array)
# # Iterate through the keypoint data
# for frame_data in keypoint_array:
#     print(frame_data)