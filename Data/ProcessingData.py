import os
from PIL import Image
import numpy as np

# Define the path to your dataset folder
p="train"
dataset_folder = fr"DataSets\SplitData\{p}"

# Initialize empty lists to store images and labels
images = []
labels = []

# Define a dictionary to map class names to numerical labels
class_to_label = {"Abnormal": 0, "Normal": 1}

# Define the target image size
target_size = (224, 224)

# Iterate through each subfolder in the dataset folder
for class_name in class_to_label.keys():
    class_folder = os.path.join(dataset_folder, class_name)
    
    # Iterate through each image file in the class folder
    for image_file in os.listdir(class_folder):
        if image_file.endswith(".png"):
            image_path = os.path.join(class_folder, image_file)
            
            # Load the image using PIL
            image = Image.open(image_path)

            # Resize the image to the target size
            image = image.resize(target_size, Image.ANTIALIAS)
            
            # Convert the image to a NumPy array
            image = np.array(image)
            image = np.transpose(image, (2, 0, 1))
            
            # Append the image and its corresponding label to the lists
            images.append(image)
            labels.append(class_to_label[class_name])

# Convert the lists to NumPy arrays
images = np.array(images,dtype=np.float32)
labels = np.array(labels,dtype=np.float32)

# Save the NumPy arrays to a file
np.savez("Data_"+p+".npz", images=images, labels=labels)
