# HDC Encoder Decoder

A High Dimensional Computing (HDC) based Encoder/Decoder for integrating with Large Language Models (LLMs) like **LLaMA**. This work demonstrates how to replace traditional embedding and output layers with HDC-based components, allowing anyone to easily test HDC with LLaMA models by modifying only a few lines of code and training on a dataset. It also includes implementations from Entropic paper and how that can be extended to HDC space.

## Table of Contents

- [Introduction](#introduction)
- [What is High Dimensional Computing (HDC)?](#what-is-high-dimensional-computing-hdc)
  - [HDC vs. Traditional Embeddings](#hdc-vs-traditional-embeddings)
  - [Benefits of HDC](#benefits-of-hdc)
- [Work Overview](#work-overview)
- [Features](#features)
- [License](#license)

## Introduction

This work implements a High Dimensional Computing (HDC) based Encoder and Decoder for Transformer-based language models like **LLaMA**. By leveraging HDC, the work showcases how to:

- Replace traditional dense embeddings with high-dimensional sparse representations.
- Integrate HDC into existing models with minimal code changes.
- Train the modified model on custom datasets.

## What is High Dimensional Computing (HDC)?

High Dimensional Computing, also known as Hyperdimensional Computing, is a computational paradigm inspired by the way the human brain might represent information using patterns of neural activity distributed across large populations of neurons.

In HDC:

- **Representations** are high-dimensional (e.g., 10,000 dimensions), sparse, and often binary or ternary.
- **Information Encoding** leverages the mathematical properties of high-dimensional spaces to encode, store, and manipulate data.
- **Operations** are typically simple (e.g., binding, bundling), yet powerful due to the properties of high-dimensional spaces.

### HDC vs. Traditional Embeddings

| Aspect                | Traditional Embeddings           | High Dimensional Computing      |
|-----------------------|----------------------------------|---------------------------------|
| **Dimensionality**    | Low-dimensional (e.g., 300)      | High-dimensional (e.g., 10,000) |
| **Representation**    | Dense, continuous vectors        | Sparse, binary/ternary vectors  |
| **Focus**             | Semantic similarity              | Robust encoding and computation |
| **Operations**        | Linear algebra (dot products)    | Simple operations (XOR, sum)    |
| **Noise Robustness**  | Susceptible to noise             | Inherently robust               |

### Benefits of HDC

- **Robustness to Noise**: High-dimensional representations are less affected by noise due to the distributed nature of the encoding.
- **Efficiency**: Operations on sparse vectors can be computationally efficient, especially with hardware acceleration.
- **Scalability**: High-dimensional spaces can represent vast amounts of information, suitable for complex tasks.

## Work Overview

This work allows anyone to easily test HDC with a LLaMA model by:

- **Replacing Few Lines of Code**: Swap out traditional embedding and output layers with HDC-based components.
- **Minimal Changes to Architecture**: The core Transformer architecture remains intact.
- **Training on Custom Datasets**: Use your own data to train and evaluate the model's performance.

## Features

- **HDC Encoder**: Converts input tokens into high-dimensional sparse vectors, replacing the embedding layer.
- **HDC Decoder**: Decodes high-dimensional representations back into tokens, replacing the output layer.
- **Integration with LLaMA Transformer**: Utilizes the existing Transformer architecture for sequence modeling.

## How It Works

### Integrating HDC with LLaMA

The integration involves:

1. **HDC Encoder**:

   - Replaces the traditional embedding layer.
   - Maps each token to a high-dimensional sparse vector.
   - Incorporates positional information using binding operations.

2. **Transformer Layers**:

   - Remain largely unchanged.
   - Process the high-dimensional representations.

3. **HDC Decoder**:

   - Replaces the output projection layer.
   - Decodes high-dimensional vectors back to token indices using similarity measures (e.g., dot product).

### Modifying Existing Models

To integrate HDC into other models:

- **Replace Embedding Layer**: Swap out the embedding layer with the `HDCEncoder`.
- **Replace Output Layer**: Replace the output projection layer with the `HDCDecoder`.
- **Adjust Input/Output Shapes**: Ensure the model handles the high-dimensional inputs and outputs correctly.
- **Training**: Fine-tune the modified model on your dataset.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
