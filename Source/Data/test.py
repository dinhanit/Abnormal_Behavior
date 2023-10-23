from Pre_Image import get_landmark_from_image
import cv2 

n_img = get_landmark_from_image("image_6.jpg")
a_img = get_landmark_from_image("image_8.jpg")

print(n_img)
print("a",a_img)
print("ok", n_img - a_img)
