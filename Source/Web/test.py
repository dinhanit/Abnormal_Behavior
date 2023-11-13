import cv2
import time
from datetime import datetime

cap = cv2.VideoCapture(0)
i = 0
while True:
    success, img = cap.read()
    # Get the current date and time
    current_time = datetime.now()

    # Print the current time
    print("Current time:", current_time)
    if success:
        i += 1
        print(i)
        cv2.imshow("image", img)
        cv2.waitKey