import cv2
import time
import os
""" 
This file contains the collect images task 

    Parameters:
     -num_images_taken: Input numbers of images you've already taken
     -key: Input "n" to take Normal Image or "a" for Abnormal Image
        for every single image, while code is running

     Returns:
     - Images to Origin folder 
"""
if not os.path.exists("Origin"):
    os.makedirs("Origin")

# Input the number of images you've already taken
num_images_taken = int(input("Enter the number of images you've already taken: "))

# Open a connection to the webcam (0 is usually the built-in webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    current_index = num_images_taken

    cv2.imshow("Captured Image", frame)  # Display the captured image

    key = cv2.waitKey(1) & 0xFF

    if key == ord("n"):
        if not os.path.exists("Origin/Normal"):
            os.makedirs("Origin/Normal")
        image_name = os.path.join("Origin", "Normal", f"image_{current_index}.jpg")
        cv2.imwrite(image_name, frame)
        num_images_taken += 1
        print(f"Saved as {image_name}")

    elif key == ord("a"):
        if not os.path.exists("Origin/Abnormal"):
            os.makedirs("Origin/Abnormal")
        image_name = os.path.join("Origin", "Abnormal", f"image_{current_index}.jpg")
        cv2.imwrite(image_name, frame)
        num_images_taken += 1
        print(f"Saved as {image_name}")

    elif key == 27:  # Press 'Esc' to exit the loop
        break

# Release the camera and close the display window when done
cap.release()
cv2.destroyAllWindows()


