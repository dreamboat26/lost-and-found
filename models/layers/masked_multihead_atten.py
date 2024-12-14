"""
 # @ Author: Meet Patel
 # @ Create Time: 2024-07-06 23:16:50
 # @ Modified by: Meet Patel
 # @ Modified time: 2024-07-12 21:56:24
 # @ Description:
 """

import torch
import torch.nn as nn

from models.layers.generate_qkv import QKVGen


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, drop_prob: float):
        """
        Generates masked multi head attention layer.

        Args:
            emb_dim (int): Embedding dimension for input tensor.
            num_heads (int): Number of attention heads.
            drop_prob (float): Dropout probability for multi head attention and
                feed forward layer.
        """
        super().__init__()
        self.qkv_gen = QKVGen(emb_dim, num_heads)
        self.MAX_NEG = torch.tensor(float("-inf"))
        emb_dim_per_head = emb_dim // num_heads
        self.sqrt_d = torch.sqrt(torch.tensor(emb_dim_per_head))
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.softmax_layer = nn.Softmax(dim=-1)
        self.dropout_layer = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward function which takes input tokens and attention mask.

        Args:
            x (torch.Tensor): Inputs obtained after positional embedding layers.
                It is of shape [batch, seq, emb_dim].
            attn_mask (torch.Tensor): Attention mask of shape [batch, 1, seq, seq].

        Returns:
            torch.Tensor: Output of Masked Multi head attention. It is of shape
                [batch, seq, emb_dim].
        """
        q, k, v = self.qkv_gen(x)  # [batch, num_head, seq, emb_dim_per_head]

        k_t = torch.transpose(
            k, 3, 2
        )  # [batch, num_head, emb_dim_per_head, seq]
        q_k_mm = torch.matmul(q, k_t)  # [batch, num_head, seq, seq]
        causal_mask = torch.triu(torch.ones_like(q_k_mm), diagonal=1).to(
            torch.bool
        )
        self.MAX_NEG = self.MAX_NEG.to(causal_mask.device)
        causal_mask = torch.where(causal_mask == True, self.MAX_NEG, 0)
        q_k_mm += causal_mask  # [batch, num_head, seq, seq]
        q_k_mm += attn_mask  # [batch, num_head, seq, seq]
        q_k_mm /= self.sqrt_d  # [batch, num_head, seq, seq]
        q_k_mm = self.softmax_layer(q_k_mm)  # [batch, num_head, seq, seq]
        q_k_mm = self.dropout_layer(q_k_mm)  # [batch, num_head, seq, seq]
        qkv_mm = torch.matmul(
            q_k_mm, v
        )  # [batch, num_head, seq, emb_dim_per_head]
        qkv_mm = torch.transpose(
            qkv_mm, 2, 1
        )  # [batch, seq, num_head, emb_dim_per_head]
        qkv_mm = qkv_mm.reshape(
            qkv_mm.shape[0], qkv_mm.shape[1], qkv_mm.shape[2] * qkv_mm.shape[3]
        )
        # [batch, seq, emb_dim]

        res = x + self.dropout_layer(qkv_mm)  # [batch, seq, emb_dim]
        res = self.layer_norm(res)  # [batch, seq, emb_dim]
        return res
