import cv2
import time
import os
""" 
This file contains the collect images task 

    Parameters:
     -num_images_taken: Input numbers of images you've already taken
     -labels_input: Input "1" to take Normal Image or "0" for Abnormal Image and code will take image automatically
     -You can specify the number of images you want to take in "num_images" variables.

    Returns:
     - Images to Origin folder 
"""
if not os.path.exists("Origin"):
    os.makedirs("Origin")

# Input labels: 0 (Abnormal) or 1 (Normal)
labels_input = input("Enter labels: 0 (Abnormal) or 1 (Normal): ")

# Determine the starting index based on existing images in the directory
if labels_input == "1":
    if not os.path.exists("Origin/Normal"):
        os.makedirs("Origin/Normal")
    # image_dir = "Origin/Normal"
else:
    if not os.path.exists("Origin/Abnormal"):
        os.makedirs("Origin/Abnormal")
    # image_dir = "Origin/Abnormal"

# Input the number of images you've already taken
num_images_taken = int(input("Enter the number of images you've already taken: "))

# existing_images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
# if existing_images:
#     last_image_index = max(int(image.split("_")[1].split(".")[0]) for image in existing_images)
# else:
#     last_image_index = -1

# Open a connection to the webcam (0 is usually the built-in webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    frame_rate = 10  # Capture 10 frames per second
    num_images = 15  # Capture 50 additional images

    for i in range(num_images):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        current_index = num_images_taken + i
        
        if labels_input == "1":
            image_name = os.path.join("Origin", "Normal", f"image_{current_index}.jpg")
        else: 
            image_name = os.path.join("Origin", "Abnormal", f"image_{current_index}.jpg")
        
        cv2.imshow("Captured Image", frame)  # Display the captured image
        cv2.imwrite(image_name, frame)
        time.sleep(1 / frame_rate)  # Sleep to control the frame rate
        cv2.waitKey(1)  # Refresh the display

    # Release the camera and close the display window when done
    cap.release()
    cv2.destroyAllWindows()

print("Done")
