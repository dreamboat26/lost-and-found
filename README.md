# Road Signs Bounding Box Prediction using PyTorch

## Overview

This project aims to predict bounding boxes around road signs in images and classify the type of road sign using PyTorch. The dataset used for this task is the [Road Signs Dataset](https://makeml.app/datasets/road-signs), which contains images of various road signs along with annotations.

## Dependencies

Ensure you have the following dependencies installed:

- `numpy`: Library for numerical operations.
- `pandas`: Library for data manipulation and analysis.
- `opencv-python`: OpenCV library for computer vision tasks.
- `torch`: PyTorch library for deep learning.
- `torchvision`: PyTorch's library for vision tasks.
- `matplotlib`: Library for data visualization.

You can install the dependencies using pip

## Dataset

The dataset consists of images and corresponding annotations in XML format. Each annotation contains information about the filename, image dimensions, class (type of road sign), and bounding box coordinates.

## Preprocessing
- Image Resizing: Images are resized to a standard size of 300x300 pixels along with resizing the bounding boxes accordingly.
- Label Encoding: The class labels are encoded numerically for classification.

## Model Architecture

The model architecture for this task involves:
- Bounding Box Prediction: A convolutional neural network (CNN) predicts the bounding box coordinates of road signs.
- Sign Type Classification: Another CNN predicts the type of road sign from the resized images.

## Training

The training process involves:
- Data Loading: Images and annotations are loaded using custom PyTorch Dataset and DataLoader.
- Model Training: The model is trained using the loaded dataset with bounding box regression and classification loss functions.
- Optimization: Adam optimizer is used for optimizing the model parameters.

## Evaluation

The trained model can be evaluated on test images to assess its performance in predicting bounding boxes and classifying road signs.

## Acknowledgments

This project utilizes the Road Signs Dataset and various libraries like OpenCV and PyTorch for implementing the bounding box prediction and classification tasks.

## License

This project is licensed under the MIT License - see the LICENSE file for details.



