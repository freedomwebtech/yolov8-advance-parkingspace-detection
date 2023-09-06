import cv2
import numpy as np
import pickle

cap = cv2.VideoCapture('easy.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame=cv2.resize(frame,(1020,500))

    cv2.imshow('Draw Polylines', frame)

    key = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()
