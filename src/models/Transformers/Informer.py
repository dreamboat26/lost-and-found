# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.Transformers.layers.masking import TriangularCausalMask, ProbMask
from src.models.Transformers.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from src.models.Transformers.layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from src.models.Transformers.layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(
                            False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

# %%
# class Configs(object):
#     ab = 0
#     seq_len = 168
#     label_len = 24
#     pred_len = 36
#     output_attention = False
#     enc_in = 12  # num features
#     dec_in = 1
#     d_model = 16
#     embed = 'timeF'
#     dropout = 0.05
#     freq = 'h'
#     factor = 1
#     n_heads = 8
#     d_ff = 16
#     e_layers = 2
#     d_layers = 1
#     c_out = 1
#     activation = 'gelu'
#     distil = False


# configs = Configs()
# %%
# model = Model(configs)

# print('parameter number is {}'.format(sum(p.numel()
#       for p in model.parameters())))
# %%

# enc = torch.randn([3, configs.seq_len, 12])
# enc_mark = torch.randn([3, configs.seq_len, 4])

# dec = torch.randn([3, 36, 1])
# dec_mark = torch.randn([3, 36, 4])

# out = model.forward(enc, enc_mark, dec, dec_mark)
# print(out.shape)
# %%
