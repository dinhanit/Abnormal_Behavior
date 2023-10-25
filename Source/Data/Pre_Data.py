import os
import pandas as pd
import numpy as np
from Pre_image_2 import get_landmark_from_image

# Define the path to your dataset folder
p="train"
dataset_folder = fr"DataSets\SplitData\{p}"

# Initialize empty lists to store images and labels
landmarks_list = []
labels = []
data = []

# Define a dictionary to map class names to numerical labels
class_to_label = {"Abnormal": 0, "Normal": 1}

# Define the target image size
target_size = (224, 224)

# Iterate through each subfolder in the dataset folder
for class_name in class_to_label.keys():
    class_folder = os.path.join(dataset_folder, class_name)
    
    # Iterate through each image file in the class folder
    for image_file in os.listdir(class_folder):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(class_folder, image_file)
            
            # Load the image using PIL
            landmark = get_landmark_from_image(image_path)
            landmark = np.array(landmark)

            # Append the data row (landmarks and label) to the list
            data.append(list(landmark) + [class_to_label[class_name]])

# Define column names for your DataFrame
column_names = [f"distance_{i}" for i in range(1, len(landmark) + 1)] + ["labels"]

# Convert the list of data to a pandas DataFrame with specified column names
df = pd.DataFrame(data, columns=column_names)

# Save the DataFrame to a CSV file
csv_filename = "Prepared_Data_" + p + ".csv"
df.to_csv(csv_filename, index=False)
