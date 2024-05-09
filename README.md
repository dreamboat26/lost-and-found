# Zero-Shot Object Detection with Grounding DINO

This project implements Zero-Shot Object Detection using Grounding DINO (Detection in Name Only) model. The model is capable of detecting objects without explicit supervision, leveraging only textual descriptions of objects during training.

## Introduction

Zero-Shot Object Detection aims to detect objects in images without the need for annotated training data for those objects. This project focuses on achieving this task using the Grounding DINO approach, which utilizes text embeddings to ground object detections.

## Dataset

The dataset used for this project is [COCO](https://cocodataset.org/), a large-scale object detection, segmentation, and captioning dataset. COCO provides images with annotations for objects in 80 categories, making it suitable for training and evaluating Zero-Shot Object Detection models.

## Project Structure

The project is organized as follows:

- `data/`: Directory containing scripts or instructions to download and preprocess the COCO dataset.
- `models/`: Implementation of the Grounding DINO model and necessary utilities.

## Getting Started

To get started with the project, follow these steps:

1. Clone this repository.
2. Install the dependencies.
3. Download and preprocess the COCO dataset.
4. Train the Grounding DINO model.
5. Evaluate the model's performance.
