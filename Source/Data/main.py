import numpy as np

# Load the data from the .npz file
data = np.load('Preprocessed_Data_test.npz', allow_pickle=True)
print(data)

# Get the names of all the arrays in the .npz file
array_names = data.files

print("Array names in the .npz file:", array_names)

# Access the keypoint array
keypoint_array = data['landmarks']
print(keypoint_array.shape)
# print(keypoint_array)
# for frame_data in keypoint_array:
#     print(frame_data)

keypoint_array = data['labels']
print(keypoint_array.shape)
# Iterate through the keypoint data
# for frame_data in keypoint_array:
#     print(frame_data)

