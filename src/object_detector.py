import cv2
import time
import threading
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, video_path: str, selected_objects: list):
        self.model = YOLO('lib/yolov8s.pt')
        self.classes = list(self.model.names.values())
        self.video_path = video_path
        self.selected_objects = selected_objects
        self.detection_running = self.detection_paused = False
        self.detection_thread = None
        self.detection_data = {}
        self.current_frame = None
        self.frames = []
        self.complete = False
    
    def start_detection(self):
        """Start the object detection process"""
        if self.detection_running:
            return
            
        self.detection_running = True
        self.detection_paused = False
        
        # Start detection in a separate thread
        self.detection_thread = threading.Thread(target=self.run_detection)
        self.detection_thread.start()
    
    def run_detection(self):
        """Run the object detection on the video"""
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        while self.detection_running:
            if self.detection_paused: # If paused
                time.sleep(0.1)
                continue
            ret, frame = cap.read()
            if not ret: # If end of video reached
                break
            frame_count += 1
            # Resize frame for processing
            frame = cv2.resize(frame, (540, 540))
            # Predict objects using YOLOv8
            results = self.model.predict(frame)
            self.detection_data[frame_count] = results
            
            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy.numpy()[0]
                    conf = box.conf
                    cls = box.cls
                    class_name = self.classes[int(cls)]
                    if class_name in self.selected_objects:
                        # Draw bounding box and label
                        color = self.selected_objects[class_name]['color']
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"{class_name}:{float(conf):.2f}", (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Update the current frame for display
            self.current_frame = frame
            self.frames.append(frame)
        self.complete = True
        cap.release()
        self.stop_detection()
    
    def pause_detection(self):
        """Pause the object detection process"""
        self.detection_paused = True
    
    def resume_detection(self):
        """Resume the object detection process"""
        self.detection_paused = False

    def is_running(self) -> bool:
        """Check if the detection is running"""
        return self.detection_running
    
    def is_paused(self) -> bool:
        """Check if the detection is paused"""
        return self.detection_paused

    def is_complete(self) -> bool:
        """Check if the detection is complete"""
        return self.complete

    def stop_detection(self):
        """Stop the object detection process"""
        self.detection_running = False
        if self.detection_thread is not None:
            self.detection_thread = None
        self.current_frame = None

if __name__ == "__main__":
    video_path = 'lib/trimmed2.mp4'
    selected_objects = {
        'car': {'color': (0, 255, 0)},
        'truck': {'color': (0, 0, 255)},
        'airplane': {'color': (255, 255, 0)},
    }
    
    detector = ObjectDetector(video_path, selected_objects)
    detector.start_detection()
    
    while True:
        if detector.current_frame is not None:
            cv2.imshow('Object Detection', detector.current_frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            detector.stop_detection()
            break
        elif key == ord('p'):
            if detector.is_paused():
                detector.resume_detection()
            else:
                detector.pause_detection()
    
        if detector.is_complete():
            print("Detection complete.")
            break

    cv2.destroyAllWindows()