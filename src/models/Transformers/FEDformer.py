# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.Transformers.layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from src.models.Transformers.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from src.models.Transformers.layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from src.models.Transformers.layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from src.models.Transformers.layers.SelfAttention_Family import FullAttention, ProbAttention
from src.models.Transformers.layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
# import math
# import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        if configs.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=configs.modes,
                                                  ich=configs.d_model,
                                                  base=configs.base,
                                                  activation=configs.cross_activation)
        else: # Fourier
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=configs.modes,
                                                      mode_select_method=configs.mode_select)
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        dec_modes = int(
            min(configs.modes, (configs.seq_len//2+configs.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        """
        The function starts by performing a seasonal decomposition on the encoder input tensor x_enc using a moving average method. 
        It then uses the resulting seasonal and trend components to initialize the decoder input tensor x_dec.

        The encoder and decoder inputs are then passed through their respective embedding layers (enc_embedding and dec_embedding), 
        which convert the inputs to a higher-dimensional space, while discarding position embeddings, as the input data already has 
        the sequential information.

        The encoder embedding is then fed to the encoder object, which is an instance of the Encoder class, consisting of EncoderLayer blocks. 
        Each EncoderLayer block consists of a AutoCorrelationLayer, which applies an attention mechanism on the input, followed by a feedforward 
        neural network layer.

        Similarly, the decoder embedding is fed to the decoder object, which is an instance of the Decoder class, consisting of DecoderLayer blocks. 
        Each DecoderLayer block consists of two AutoCorrelationLayers, one that applies self-attention on the decoder input, and another that applies 
        cross-attention on the encoder output, followed by a feedforward neural network layer.

        Finally, the decoder output is obtained by adding the seasonal and trend components generated by the decoder, and returned in the shape 
        [batch_size, pred_len, c_out], where c_out is the number of outputs. If output_attention is set to True during model initialization, 
        the function also returns the attention weights from the decoder.
        """
        # batch_size = x_enc.shape[0]
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                            x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(
            seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# %%


# class Configs(object):
#     ab = 0
#     modes = 32
#     mode_select = 'random'
#     # version = 'Fourier'
#     version = 'Wavelets'
#     moving_avg = [12, 24]
#     L = 1
#     base = 'legendre'
#     cross_activation = 'tanh'
#     seq_len = 96
#     label_len = 48
#     pred_len = 96
#     output_attention = True
#     enc_in = 7
#     dec_in = 7
#     d_model = 16
#     embed = 'timeF'
#     dropout = 0.05
#     freq = 'h'
#     factor = 1
#     n_heads = 8
#     d_ff = 16
#     e_layers = 2
#     d_layers = 1
#     c_out = 7
#     activation = 'gelu'
#     wavelet = 0


# configs = Configs()

# # # %%

# model = Model(configs)
# # %%
# print('parameter number is {}'.format(sum(p.numel()
#       for p in model.parameters())))
# enc = torch.randn([3, configs.seq_len, 7])
# enc_mark = torch.randn([3, configs.seq_len, 4])

# dec = torch.randn([3, configs.seq_len//2+configs.pred_len, 7])
# dec_mark = torch.randn([3, configs.seq_len//2+configs.pred_len, 4])
# out = model.forward(enc, enc_mark, dec, dec_mark)
# print(out)
