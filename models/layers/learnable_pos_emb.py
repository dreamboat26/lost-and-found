import torch
import torch.nn as nn


class LearnablePositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len: int, emb_dim: int):
        """
        Generates Learnable Positional Embeddings.

        Args:
            max_seq_len (int): Maximum sequence length for which positional
                embeddings are to be trained.
            emb_dim (int): Embedding dimension for positional embeddings.
        """
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, emb_dim))
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function of positional embeddings.

        Args:
            x (torch.Tensor): Input tensor which is obtained after embedding
                layer. It is of shape [batch, seq, emb_dim].

        Returns:
            torch.Tensor: Output tensor which is sum of input tensor and
                positional embeddings. It is of shape [batch, seq, emb_dim].
        """
        batch_seq_len = x.shape[1]
        assert (
            batch_seq_len <= self.max_seq_len
        ), "Sequence length of the batch is more than max sequence length."
        pos_emb_value = self.pos_emb[:, :batch_seq_len, :]
        x = x + pos_emb_value  # [batch, seq, emb_dim]
        return x
