import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from tracker import *  # Ensure the Tracker class is defined correctly in tracker.py
import numpy as np

# Load the YOLO model
model = YOLO('lib/yolov8s.pt')

# Function to print the mouse position in RGB window
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

# Create a named window and set a mouse callback function
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture('lib/trimmed.mp4')  # Initialize video capture with the video file

# Open the 'coco.txt' file containing class names and read its content
with open("lib/coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")  # Split the content by newline to get a list of class names

# Initialize counters and trackers
count = 0
car_count = 0
bus_count = 0
truck_count = 0
tracker = Tracker()
cy1 = 184
cy2 = 209
offset = 8

# Create class dict
classes = {}
for class_ in class_list:
    classes[class_] = {
                    'color':tuple(np.random.randint(0,255,(1,3)).tolist()[0])
                    }

# Start processing the video frame by frame
while True:
    for class_ in class_list:
        classes[class_]['coords'] = []
        classes[class_]['bbox'] = []
        classes[class_]['count'] = 0
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:  # If no frame is read (end of video), break the loop
        break
    count += 1  # Increment frame count
    if count % 3 != 0:  # Process every third frame
        continue
    frame = cv2.resize(frame, (1020, 500))  # Resize the frame for consistent processing

    # Predict objects in the frame using YOLO model
    results = model.predict(frame)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")  # Convert the prediction results into a pandas DataFrame

    # Iterate over the detection results and categorize them into cars, buses, or trucks
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        classes[c]['coords'].append([x1, y1, x2, y2])  # Append the bounding box coordinates to the respective class list

    # Update tracker for each vehicle type
    for key in classes:
        classes[key]['bbox'] = tracker.update(classes[key]['coords'])

    # Check each car, bus, and truck
    for class_, value in classes.items():
        for bbox in value['bbox']:
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            if (cy > cy1 - offset) and (cy < cy1 + offset):
                classes[class_]['count'] += 1
        
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), classes[class_]['color'], 2)
            cvzone.putTextRect(frame, '{class_}'.format(class_=class_), (bbox[0], bbox[1]), 1, 1, colorR=classes[class_]['color'])
        # # Print the total count for each vehicle type
        # print('Total {class_} count: {count}'.format(class_=class_, count=classes[class_]['count']))
    
    # Display the frame in the 'RGB' window
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Break the loop if 'Esc' key is pressed
        break

# Release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()