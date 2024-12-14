"""
 # @ Author: Daniel Lin
 # @ Create Time: 2024-07-06 22:43:40
 # @ Modified by: Daniel Lin
 # @ Modified time: 2024-07-08 21:56:31
 # @ Description:
 """

import torch
import torch.nn as nn

from models.helper import Config
from models.layers.decoder_block import TransformerDecoderBlock
from models.layers.learnable_pos_emb import LearnablePositionalEmbeddings


class GPTModel(nn.Module):
    def __init__(self, config: Config):
        """
        Creates a GPT model from given config.

        Args:
            config (Config): Instance of Config which contains model
                architecture related parameters.
        """
        super().__init__()
        self.emb_layer = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb_layer = LearnablePositionalEmbeddings(
            config.max_seq_len, config.emb_dim
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.emb_dim,
                    config.num_heads,
                    config.ff_multiplier,
                    config.drop_prob,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.MAX_NEG = torch.tensor(float("-inf"))
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size)
        self.lm_head.weight = nn.Parameter(
            self.emb_layer.weight
        )  # sharing of weights between embedding layer and language model head

    def update_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Update the attention mask from [batch, seq] into [batch, 1, seq, seq].
        During this the mask will undergo transformations to make if usable in
        attention block.

        Args:
            mask (torch.Tensor): attention mask input tensor of shape [batch, seq].

        Returns:
            torch.Tensor: Updated attention mask of shape [batch, 1, seq, seq].
        """
        mask = mask.to(torch.float32)
        mask = mask.view(
            mask.shape[0], 1, 1, mask.shape[1]
        )  # [batch, 1, 1, seq]
        mask = 1 - mask  # invert the mask
        self.MAX_NEG = self.MAX_NEG.to(mask.device)
        mask = torch.where(mask == 0, 0, self.MAX_NEG)
        return mask  # [batch, 1, 1, seq]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward function for GPT model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq].
            mask (torch.Tensor): Attention mask of shape [batch, seq].

        Returns:
            torch.Tensor: Output of GPT model of shape [batch, seq, vocab_size].
        """
        x = self.emb_layer(x)  # [batch, seq, emb_dim]
        x = self.pos_emb_layer(x)  # [batch, seq, emb_dim]
        mask = self.update_mask(mask)  # [batch, 1, 1, seq]
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)  # [batch, seq, emb_dim]

        x = self.lm_head(x)  # [batch, seq, vocab_size]
        return x
