import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, seq_len, pred_len, enc_in=7, individual=False):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = enc_in
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        self.outlinear = nn.Linear(12*167, 12)

    def forward(self, x):
        batch_size = x.shape[0]
        # print(x.shape)
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros(
                [x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
            x = x.reshape(batch_size, 12*167).contiguous()
            out = self.outlinear(x)
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
            # print(x.shape)

            x = x.reshape(batch_size, 12*167).contiguous()
            # print(x.shape)
            out = self.outlinear(x)
        return out  # [Batch, Output length, Channel]


# x = torch.randn(64, 168, 495)
# model = Model(seq_len=168, pred_len=12, enc_in=7, individual=False)
# out = model(x)
# out.shape
# == torch.Size([64, 12, 170])
