## Object Tracking and Camera Movement Demo with Norfair

This repository contains a demo showcasing object tracking and camera movement tracking using Norfair, similar to the Hugging Face Spaces Norfair demo.

### Overview

The demo utilizes the YOLOv7 model for object detection and Norfair for object tracking and camera movement tracking. It processes frames of a video, detects objects in each frame, tracks the movement of those objects, and also tracks the camera movement to improve object tracking accuracy.

### Instructions

1. **Setup Environment**: Ensure you have the required dependencies installed, including Norfair, YOLOv7, and any other necessary libraries.

2. **Load Video**: Load the video you want to analyze. The default video provided in the demo can be used, or you can specify a different one.

3. **Initialize YOLOv7 Model**: Initialize the YOLOv7 model for object detection.

4. **Process Frames**: Process each frame of the video using the YOLOv7 model to detect objects and track them using Norfair.

5. **Draw Object Paths**: Draw the paths of the detected objects on the video frames.

6. **Track Camera Movement**: Utilize Norfair's camera movement tracking feature to improve object tracking and keep object paths fixed despite camera movements.

7. **Display Results**: Display the processed video with object paths and camera movement tracking.

### Usage

\```python
import cv2
import norfair
from yolov7 import YOLOv7

# Load the video
video_url = "https://example.com/video.mp4"
video = cv2.VideoCapture(video_url)

# Initialize YOLOv7 model
yolo = YOLOv7()

# Initialize Norfair tracker
tracker = norfair.Tracker()

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Object detection with YOLOv7
    detections = yolo.detect(frame)

    # Update tracker with new detections
    tracker.update(detections)

    # Draw object paths
    frame = tracker.draw_paths(frame)

    # Track camera movement
    tracker.update_camera_movement()

    cv2.imshow('Object Tracking Demo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


