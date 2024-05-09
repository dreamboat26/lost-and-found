# Depth Estimation with DINO-2

## Introduction
This project implements depth estimation using DINO-2 (Depth In Name Only), a variant of the DINO model adapted for depth estimation tasks. DINO-2 leverages self-supervised learning to learn depth representations from monocular images.

## Model Architecture
The DINO-2 model architecture consists of a neural network backbone followed by a depth prediction head. The backbone is typically a vision transformer trained with a contrastive learning objective, while the depth prediction head predicts the depth map from the learned features.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib

## Installation
1. Clone this repository:
2. Install the required dependencies

## Dataset
The dataset used for depth estimation consists of pairs of RGB images and corresponding depth maps. Common datasets for depth estimation include NYU Depth, KITTI, and the Make3D dataset.

## Results
The results of training and evaluation will be displayed in the console. Evaluation metrics will be reported.

## References
- [Paper: DINO: Vision Transformers with Self-Supervised Learning](https://arxiv.org/abs/2104.14294)
- [PyTorch](https://pytorch.org/)
- [NYU Depth Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [Make3D Dataset](http://make3d.cs.cornell.edu/data.html)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

