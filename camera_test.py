import cv2

cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())
if cap.isOpened():
    ret, frame = cap.read()
    print("Frame read:", ret)
    cap.release()
else:
    print("Camera could not be opened")