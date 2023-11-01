from keyPointDetect import DetectKeyPoint
import torch
import torch.nn.functional as F
import cv2
from param import DEVICE

model = torch.load(".model/Weight").to(DEVICE)

def Inference(img):
    global model
    label = ['Abnormal', 'Normal']
    kp = DetectKeyPoint(img)
    kp = kp.reshape(1, -1)
    model.eval()
    with torch.no_grad():
        output = model(torch.Tensor(kp).to(DEVICE))
    probabilities = F.softmax(output, dim=1)
    if probabilities[0][1] < 0.001:
        predicted_class = 0
    else:
        predicted_class = 1
    return label[predicted_class]

font_color = {"Normal": (0, 255, 0),
              "Abnormal": (0, 0, 255)}

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
font_scale = 1
thickness = 2

# Change the 'video_file' variable to the path of your video file
video_file = 'video_test2.mp4'
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

true_predicted = 0
frame_count = 0  # Initialize frame count


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1  # Increment the frame count
    label = Inference(frame)
    frame = cv2.putText(frame, label, org, font, font_scale, font_color[label], thickness, cv2.LINE_AA)
    cv2.imshow('Video Stream', frame)
    key = cv2.waitKey(0)

    # If true frame is Abnormal then press "1"
    if key == ord('1') and label == "Abnormal":
        true_predicted += 1
    # If true frame is Normal then press "0"
    elif key == ord('0') and label == "Normal":
        true_predicted += 1
    #press "q" to exit
    elif key & 0xFF == ord('q'):
        break
    else:
        pass
cap.release()
cv2.destroyAllWindows()

print("Total Frames Processed:", frame_count)  # Print the total frame count
print("True Predictions:", true_predicted)

performance = true_predicted / frame_count
print("Performance (Accuracy): {:.2f}%".format(performance * 100))

