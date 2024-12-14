"""
 # @ Author: Meet Patel
 # @ Create Time: 2024-07-07 00:23:15
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-07-11 21:25:59
 # @ Description:
 """

import torch
import torch.nn as nn


class FFLayer(nn.Module):
    def __init__(self, emb_dim: int, ff_multiplier: int, drop_rate: float):
        """
        Generate FeedForward layer block of Transformer layer.

        Args:
            emb_dim (int): Embedding dimension for input tensor.
            ff_multiplier (int): Feed forward layer dimensionality multiplier.
            drop_prob (float): Dropout probability for multi head attention and
                feed forward layer.
        """
        super().__init__()
        self.ff1 = nn.Linear(emb_dim, emb_dim * ff_multiplier)
        self.ff2 = nn.Linear(emb_dim * ff_multiplier, emb_dim)
        self.gelu = nn.GELU()
        self.dropout_layer = nn.Dropout(p=drop_rate)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for Feedforward layer.

        Args:
            x (torch.Tensor): Input to feedforward layer. It will be of shape
                [batch, seq, emb_dim].

        Returns:
            torch.Tensor: Output of feedforward layer. It will be of shape
                [batch, seq, emb_dim]
        """
        res = self.dropout_layer(
            self.gelu(self.ff1(x))
        )  # [batch, seq, emb_dim * ff_multiplier]
        res = self.ff2(res)  # [batch, seq, emb_dim]
        res = (
            self.dropout_layer(self.layer_norm(res)) + x
        )  # [batch, seq, emb_dim]
        return res
