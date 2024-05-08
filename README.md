# Planar Arm Control and Face Landmark Detection

## Overview

This repository contains code for two distinct functionalities:

1. **Planar Arm Control**:
   - Control and visualization of a planar arm with an arbitrary number of links.
   - Implementation of collision detection with obstacles in the environment.
   - Pathfinding using the A* algorithm on a toroidal grid.

2. **Face Landmark Detection**:
   - Loading a pre-trained neural network for detecting facial landmarks.
   - Evaluation of the model on test images.
   - Visualization of the detected landmarks alongside ground truth landmarks.

## Planar Arm Control

### Functionality
- The `NLinkArm` class facilitates control and plotting of a planar arm.
- It supports an arbitrary number of links and joint angles.
- Collision detection with obstacles is implemented to ensure safe arm movements.
- Pathfinding using the A* algorithm on a toroidal grid enables finding obstacle-free routes.

### Usage
- Modify the `main()` function to set arm configurations, start, and goal positions.
- Execute the script to visualize arm movements and pathfinding results.

## Face Landmark Detection

### Functionality
- Pre-trained neural network for detecting facial landmarks is loaded and evaluated.
- Test images are processed, and predicted landmarks are visualized alongside ground truth landmarks.

### Usage
- Ensure the pre-trained model file (`face_landmarks.pth`) is available.
- Execute the script to load and evaluate the model on test images.
- Visualize the detected landmarks for evaluation.

## Dependencies

Ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- Matplotlib
- NumPy

## Acknowledgments

The code for collision detection in the `detect_collision` function is adapted from [Doswa's Circle Segment Intersection](http://doswa.com/2009/07/13/circle-segment-intersectioncollision.html).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


