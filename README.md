# Mixture of Experts (MoE) with Switch Transformers 

## Introduction

This notebook introduces the first MoE model developed by Google AI, as presented in the paper "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity".

The architecture of the MoE model bears resemblance to T5 models, with the FeedForward layer of the attention layer being slightly different. In the encoder and decoder model, the latest layer is replaced by a "Sparse MLP". The figure below is extracted from the original paper.

## Experiment Overview

### Purpose
The purpose of this experiment is to understand and implement the Mixture of Experts (MoE) model as described in the paper.

### Methodology
1. **Model Architecture Review**: Review the architecture details provided in the paper.
2. **Implementation**: Implement the MoE model using TensorFlow or PyTorch.
3. **Training**: Train the MoE model on a suitable dataset.
4. **Evaluation**: Evaluate the performance of the trained model.

## Model Architecture

The MoE model architecture shares similarities with T5 models but differs in the implementation of the FeedForward layer of the attention layer and the replacement of the latest layer with a "Sparse MLP" in both the encoder and decoder.

### Figure: Model Architecture (Extracted from the Paper)

![Untitled](https://github.com/dreamboat26/lost-and-found/assets/125608791/de7e6cde-cf3b-4335-b1e2-4f183fa60e3b)


## Conclusion

This experiment serves as a foundational step towards understanding and implementing the MoE model. The insights gained from this experiment can pave the way for further exploration and research in the field of large-scale language models.

For detailed implementation and results, refer to the corresponding sections in the notebook.

## References

- Original Paper: [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)

