__author__ = "Valerian Gregoire--Begranger, Maeva Jalama"
import cv2
import numpy as np
from ultralytics import YOLO

if __name__ == '__main__':
    # Load the YOLO model
    model = YOLO('lib/yolov8s.pt')

    win_name = 'ML Project (YOLOv8) - V. GREGOIRE--BEGRANGER, M. JALAMA'
    cv2.namedWindow(win_name)

    cap = cv2.VideoCapture('lib/st_bart_landing.mp4')

    # Get the width and height of the video frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

    # Create class dict with random colors
    class_list = list(model.names.values())
    classes = {
        class_: {
            'color': tuple(np.random.randint(0, 255, (1, 3)).tolist()[0])
        }
        for class_ in class_list
    }

    # Start processing the video frame by frame
    count = 0
    while True:
        # Clear previous frame's detections
        for class_ in class_list:
            classes[class_]['coords'] = []

        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 2 != 0:
            continue  # Skip odd frames

        frame = cv2.resize(frame, (width, height))

        # YOLOv8 prediction
        results = model.predict(frame, verbose=False)
        boxes = results[0].boxes

        # Populate class coordinates
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = class_list[cls_id]
            classes[class_name]['coords'].append([x1, y1, x2, y2])

        # Draw bounding boxes and labels
        for class_, data in classes.items():
            color = data['color']
            for bbox in data['coords']:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label background rectangle
                label = class_
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w + 4, y1), color, -1)

                # Put class label
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
