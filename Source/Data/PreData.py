import os
import pandas as pd
import numpy as np
from PreImage2 import get_landmark_from_image

def PreData(p):# Define the path to your dataset folder
    '''
    This function will get all images from train/test in SpitData folder and take landmarks

    Parameters:
     -p: path to your dataset folder (train or test)

    Returns:
     - CSV data frame
    '''
    p=p
    dataset_folder = fr"DataSets\SplitData\{p}"

    # Initialize empty lists to store images and labels
    data = []

    # Define a dictionary to map class names to numerical labels
    class_to_label = {"Abnormal": 0, "Normal": 1}

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
    csv_filename = "PreparedData_" + p + ".csv"
    df.to_csv(csv_filename, index=False)

PreData("train")
PreData("test")
print("Preprocess all data done")

