# YOLOv8 Object Detection with Tkinter Interface

This project demonstrates the use of the Ultralytics YOLOv8 object detection model on video files, leveraging both command-line and graphical user interface (GUI) approaches. The objective is to detect multiple object classes defined in the COCO dataset using pre-trained weights.

## Overview

The application includes three main components:

### 1. `non_threading_CLI_YOLO.py`

This script provides a simple, command-line-based object detection demonstration using YOLOv8. It loads a video file, processes it frame by frame, and draws bounding boxes for detected objects. It does not use multithreading and is suitable for testing the model output directly through OpenCV without user interaction or interface complexity.

Key characteristics:
- Synchronous execution
- Minimal dependencies (OpenCV and Ultralytics)
- Random colors assigned to object classes
- Every second frame is processed for efficiency

### 2. `ObjectDetector.py`

This module implements a multithreaded object detection backend using YOLOv8. It encapsulates video processing in a class, allowing for pause, resume, and stop functionalities. It is intended to be integrated into GUI applications and supports real-time interaction without freezing the user interface.

Key characteristics:
- Runs detection in a background thread
- Can be paused and resumed dynamically
- Tracks frame-level results and provides annotated output
- Uses a selected subset of COCO classes with custom colors

This script can also be executed independently to test threaded detection functionality in an OpenCV window.

### 3. `app.py`

This is a Tkinter-based graphical interface built around the `ObjectDetector` class. It provides users with a modern GUI to load videos, select which object classes to detect, and assign colors for bounding boxes. It supports starting, pausing, and stopping detection interactively.

Key characteristics:
- Object selection via checkboxes
- Color picker for customizing bounding box appearance
- Live video preview and statistics area
- Integration with the `ObjectDetector` thread-based backend
- Responsive interface with consistent dark-themed design

## Requirements

The following Python packages are required:

- `ultralytics`
- `opencv-python`
- `Pillow`
- `tkinter`

Install dependencies with:

```bash
pip install ultralytics opencv-python Pillow
