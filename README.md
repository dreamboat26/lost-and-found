# Optical Flow and Image Stitching

## Overview

This repository contains Python scripts for optical flow analysis and image stitching using OpenCV.

### Optical Flow Analysis

The script processes a video file (`test8.mp4`) and performs the following steps:

- Reads the video file frame by frame.
- Calculates optical flow using the Lucas-Kanade method.
- Estimates the displacement of keypoints between frames.
- Adjusts the frame size based on the detected movement to handle changes in perspective.

### Image Stitching

The script stitches two input images (`merge1.jpg` and `merge2.jpg`) together to create a panoramic image. The stitching process involves:

- Detecting keypoints and extracting local invariant descriptors from the input images.
- Matching features between the two images.
- Computing the homography matrix to align the images.
- Warping and blending the images to create a seamless panorama.

## Dependencies

Ensure you have the following dependencies installed:

- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib (for visualization)

You can install the dependencies using pip:

## Acknowledgments

This project utilizes the OpenCV library for computer vision tasks.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
