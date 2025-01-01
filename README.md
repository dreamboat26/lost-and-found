# MNIST Representation Learning with Contrastive Learning and Autoencoders

This project demonstrates the use of **self-supervised learning** methods to learn meaningful representations of the MNIST dataset using two models:
1. A **contrastive learning model** for robust representation learning.
2. An **autoencoder** for image reconstruction.

## Overview

### Contrastive Learning
The contrastive learning model uses a custom augmentation layer to generate augmented views of input images. It learns:
- **Similar representations** for augmented views of the same image.
- **Distinct representations** for views of different images.

### Autoencoder
The autoencoder reconstructs input images by compressing them into a latent representation (via the encoder) and reconstructing them using a decoder. It learns:
- To compress and reconstruct images effectively while preserving critical features.

Both models share a common **encoder**, which extracts features from images.

## Key Features
- **Self-Supervised Learning**: No labels are required for training.
- **Contrastive Loss**: Implements the InfoNCE loss to ensure augmentation-invariant representations.
- **Data Augmentation**: Includes random rotations, flips, and crops for robust feature learning.
- **t-SNE Visualization**: Visualizes the learned latent space using dimensionality reduction.

## Requirements
- Python 3.x
- TensorFlow 2.x
- TensorFlow Addons
- NumPy
- Matplotlib
- scikit-learn (for t-SNE visualization)

## How It Works
1. **Preprocessing**:
   - Normalizes and reshapes the MNIST images.
2. **Contrastive Learning**:
   - Trains the encoder using the contrastive loss with augmented views.
3. **Autoencoder**:
   - Fine-tunes the encoder and trains the decoder for image reconstruction.
4. **Evaluation**:
   - Visualizes the learned representations using t-SNE.
   - Compares original and reconstructed images.

## Code Structure
- `augment(image)`: Custom data augmentation function.
- `contrastive_loss`: Implements the InfoNCE loss.
- `AugmentationLayer`: Custom TensorFlow layer for generating augmented views.
- `autoencoder`: Model for reconstructing MNIST images.
- `contrastive_model`: Model for training the encoder with contrastive loss.

## Results
1. **Reconstruction**:
   - The autoencoder successfully reconstructs MNIST digits.
   - Visual comparison of original vs. reconstructed images is provided.
2. **Representation Learning**:
   - t-SNE visualization shows well-clustered representations of the digits.
