# DINO-DETR Final Project

This project combines the DINO (Vision Transformers trained with a Contrastive method) and DETR (Detection Transformer) models to create an end-to-end object detection system capable of zero-shot learning.

## Introduction

DINO-DETR integrates the strengths of DINO's self-supervised learning approach and DETR's object detection capabilities. This combination allows the model to detect objects in images without the need for annotated training data, leveraging only image and text embeddings.

## Sources

- [DINO: Vision Transformers with Self-Supervised Learning](https://arxiv.org/abs/2104.14294) - Paper introducing DINO.
- [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) - Paper introducing DETR.
- [Vision Transformers](https://github.com/lucidrains/vit-pytorch) - PyTorch implementation of Vision Transformers.
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Library for natural language processing and transformer models.

## Dataset

The dataset used for this project can be any standard object detection dataset, such as COCO or Pascal VOC. These datasets provide images with annotations for objects in various categories, which are used for training and evaluation.

## Project Structure

The project is organized as follows:

- `data/`: Directory containing scripts or instructions to download and preprocess the dataset.
- `models/`: Implementation of the DINO-DETR model and necessary utilities.
- `train.py`: Script for training the DINO-DETR model.
- `evaluate.py`: Script for evaluating the model's performance on the dataset.
- `requirements.txt`: List of dependencies required to run the project.

## Getting Started

To get started with the project, follow these steps:

1. Clone this repository.
2. Install the dependencies listed in `requirements.txt`.
3. Download and preprocess the dataset.
4. Train the DINO-DETR model using the `train.py` script.
5. Evaluate the model's performance using the `evaluate.py` script.

## Results

The results of the evaluation will be provided, including metrics, which measures the model's accuracy in detecting objects.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 
- Authors of the DINO and DETR papers for their groundbreaking research.
- Contributors to the object detection datasets for providing valuable data for training and evaluation.
- Maintainers of open-source libraries used in this project for enabling efficient development and experimentation.

