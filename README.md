# LCM Prediction with Chaotic Neural Network

This project uses a neural network to predict the **Least Common Multiple (LCM)** of two numbers, leveraging a **chaotic layer** based on the **Lorenz system** to enhance feature learning.

## Overview
This model predicts the **LCM** of two numbers using a **ChaoticLayer** that simulates the **Lorenz system** to generate additional complex features for more accurate predictions.

## Model Architecture

1) Chaotic Layer: Generates features using the Lorenz system (mean, std, radius).
LCMNet:
- Input: Pair of numbers.
- Hidden Layer: 16 units.
- Output: Predicted LCM value.

2) Training Process
- Loss: Mean Squared Error (MSE).
- Optimizer: RAdam with learning rate scheduler.

3) Training and Testing

The model is trained on random pairs of numbers (1-50).
After training, it evaluates the test set and prints:
- Loss
- MAPE
- Predicted vs. True LCM values
