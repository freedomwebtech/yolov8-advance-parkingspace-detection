import cv2
import numpy as np
import pickle
points = []
drawing = False

area_names = []
current_area_name = ""

cap = cv2.VideoCapture('easy.mp4')
try:
    with open('polylines.pkl', 'rb') as f:
        data = pickle.load(f)
        polylines, area_names = data['polylines'], data['area_names']
except:
    polylines=[]
def draw_polyline(event, x, y, flags, param):
    global points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(points) > 1:
            # Prompt the user for the area name
            current_area_name = input("Enter area name: ")
            if current_area_name:
                area_names.append(current_area_name)
                polylines.append(np.array(points, dtype=np.int32))
                points = []
cv2.namedWindow('Draw Polylines')
cv2.setMouseCallback('Draw Polylines', draw_polyline)
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame=cv2.resize(frame,(1020,500))

    # Draw existing polylines on the frame with their names
    for i, polyline in enumerate(polylines):
        if 'area1' in area_names:
            
            cv2.polylines(frame, [polyline], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(frame, area_names[i], tuple(polyline[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Draw Polylines', frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF

    # Save polylines and exit on 's' key press
    if key == ord('s'):
        data = {'polylines': polylines, 'area_names': area_names}
        with open('polylines.pkl', 'wb') as f:
            pickle.dump(data, f)
        break

    # Exit on 'q' key press
    if key == ord('q'):
        break

cv2.destroyAllWindows()
