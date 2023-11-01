import os
import pandas as pd
import numpy as np
from PreImage import get_landmark_from_image

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

# PreData("train")
# PreData("test")
# print("Preprocess all data done")

def PreprocessData(save_directory, train_data, test_data, train_file_name='PreprocessedData_train.npz', test_file_name='PreprocessedData_test.npz'):
    """
    Preprocess data, remove null rows, and save it to .npz files.

    Parameters:
    - save_directory: The directory where the .npz files will be saved.
    - train_data: The DataFrame for the training data.
    - test_data: The DataFrame for the testing data.
    - train_file_name: The name of the .npz file for the training data (default is 'PreprocessedData_train.npz').
    - test_file_name: The name of the .npz file for the testing data (default is 'PreprocessedData_test.npz').
    """
    def is_row_all_zero_except_label(df):
        """
        Check if all columns in the DataFrame, except the 'labels' column, are equal to 0 for each row.

        Parameters:
        df (DataFrame): The input DataFrame.

        Returns:
        (Series): A boolean Series indicating True for rows where all columns, except 'labels', are 0, and False otherwise. Rows where 'labels' may contain non-zero values.
        """
        return (df.drop('labels', axis=1) == 0).all(axis=1)

    # Check for null rows
    null_rows_train = is_row_all_zero_except_label(train_data)
    null_rows_test = is_row_all_zero_except_label(test_data)

    # Remove null rows and reset the index
    train_data = train_data[~null_rows_train].reset_index(drop=True)
    test_data = test_data[~null_rows_test].reset_index(drop=True)

    # Specify the file paths for the DataFrames
    combined_data_train_path = os.path.join(save_directory, train_file_name)
    combined_data_test_path = os.path.join(save_directory, test_file_name)

    # Save.npz file
    np.savez(combined_data_train_path, landmarks=train_data.iloc[:, :-1], labels=train_data.iloc[:, -1])
    np.savez(combined_data_test_path, landmarks=test_data.iloc[:, :-1], labels=test_data.iloc[:, -1])

save_directory = ""

# Load the data from the .csv files into DataFrames
PreparedData_train = pd.read_csv('PreparedData_train.csv')
PreparedData_test = pd.read_csv('PreparedData_test.csv')

#Call_func:
PreprocessData(save_directory, PreparedData_train, PreparedData_test)










