import cv2
import numpy as np


cap = cv2.VideoCapture()

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame=cv2.resize(frame,(1020,500))

    cv2.imshow('FRAME', frame)

    Key = cv2.waitKey(1) & 0xFF
cap.release()
cv2.destroyAllWindows()
