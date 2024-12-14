"""
 # @ Author: Meet Patel
 # @ Create Time: 2024-07-07 00:29:11
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-07-11 21:15:52
 # @ Description:
 """

import torch
import torch.nn as nn

from models.layers.ff_layer import FFLayer
from models.layers.masked_multihead_atten import MaskedMultiHeadAttention


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self, emb_dim: int, num_heads: int, ff_multiplier: int, drop_prob: float
    ):
        """
        Initializer for Transformer Decode Block. This include MaskedMultiHeadAttention
        layer and FeedForward layer.

        Args:
            emb_dim (int): Embedding dimension for input tensor.
            num_heads (int): Number of attention heads.
            ff_multiplier (int): Feed forward layer dimensionality multiplier.
            drop_prob (float): Dropout probability for multi head attention and
                feed forward layer.
        """
        super().__init__()
        self.mmha_layer = MaskedMultiHeadAttention(
            emb_dim, num_heads, drop_prob
        )
        self.ff_layer = FFLayer(emb_dim, ff_multiplier, drop_prob)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward function for Decoder block.

        Args:
            x (torch.Tensor): Input tensor for decoder layer. It will be of shape
                [batch, seq, emb_dim].
            mask (torch.Tensor): Input mask to filter out redundant tokens from
                computation. It will be of shape [batch, 1, 1, seq].

        Returns:
            torch.Tensor: Output of decoder layer. It will be of shape
                [batch, seq, emb_dim].
        """

        mmha_output = self.mmha_layer(x, mask)  # [batch, seq, emb_dim]
        ff_output = self.ff_layer(mmha_output)  # [batch, seq, emb_dim]
        return ff_output
