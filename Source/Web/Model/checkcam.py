import cv2

# Create a VideoCapture object to access the camera (usually, 0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read a frame.")
        break

    # Display the frame in a window
    cv2.imshow('Camera Stream', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
