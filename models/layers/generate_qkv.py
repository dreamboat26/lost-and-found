"""
 # @ Author: Meet Patel
 # @ Create Time: 2024-07-06 23:17:55
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-07-11 21:28:40
 # @ Description:
 """

from typing import Tuple

import torch
import torch.nn as nn


class QKVGen(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int):
        """
        Generate Q, K and V tensors from given tensor.

        Args:
            emb_dim (int): Embedding dimension for input tensor.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        assert (
            emb_dim % num_heads == 0
        ), "Embedding dimension should be divisible by Number of heads."
        self.linear = nn.Linear(emb_dim, emb_dim * 3)
        self.num_heads = num_heads

    def reshape_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape the input tensor to contain a dimension for num_heads.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq, emb_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch, num_heads, seq, emb_dim_per_head].
        """
        x = x.view(
            x.shape[0], x.shape[1], self.num_heads, x.shape[2] // self.num_heads
        )
        x = torch.transpose(x, 2, 1)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward function for Q, K and V generation.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq, emb_dim].

        Returns:
            Tuple[torch.Tensor]: Tuple of Q, K and V tensor. Each will have
                shape [batch, num_heads, seq, emb_dim_per_head].
        """
        x = self.linear(x)  # [batch, seq, emb_dim * 3]
        split_size = x.shape[2] // 3
        q, k, v = torch.split(
            x, split_size_or_sections=split_size, dim=2
        )  # [batch, seq, emb_dim]
        q = self.reshape_tensor(q)  # [batch, num_head, seq, emb_dim_per_head]
        k = self.reshape_tensor(k)  # [batch, num_head, seq, emb_dim_per_head]
        v = self.reshape_tensor(v)  # [batch, num_head, seq, emb_dim_per_head]
        return q, k, v
