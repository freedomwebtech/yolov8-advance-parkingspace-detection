import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone


with open('polylines.pkl', 'rb') as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

model=YOLO('yolov8s.pt')

# Load previously drawn polylines from a file (if available)


r=[]
def p(f,cx1,cy1):
   

    for i, polyline in enumerate(polylines):
        
        result=cv2.pointPolygonTest((polyline),((cx1,cy1)),False)
        if result>=0:
            r.append(polyline)
        cv2.polylines(frame, [polyline], isClosed=True, color=(0, 255, 0), thickness=2)
       

            

# Create a video capture object
cap = cv2.VideoCapture('easy.mp4')  # You can change 0 to the path of a video file if you want to process a video file

# Create a window for displaying the video

tracker=Tracker()

count=0
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    frame_copy = frame.copy()
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    
    list=[]


    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        if 'car'in  c:
            list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
   
    for bbox in (bbox_idx):
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        p(frame,cx,cy)

    
    for i1 in r:
        cv2.polylines(frame, [i1], isClosed=True, color=(0, 0,255), thickness=2)
    
    cv2.imshow('Video', frame)
    key = cv2.waitKey(1) & 0xFF

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
