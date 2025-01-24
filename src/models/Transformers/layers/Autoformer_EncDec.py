import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.Transformers.layers.SelfAttention_Family import FullAttention


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part

    The constructor of this class takes an integer channels as input, which represents 
    the number of channels in the input tensor. In the __init__ method, it initializes a 
    LayerNorm instance with channels as the input size.

    The forward method takes an input tensor x of shape (batch_size, sequence_length, channels)
    and applies layer normalization on the last dimension of the input tensor using the LayerNorm
    instance. The output of layer normalization is then subtracted by the mean along the second
    dimension of the normalized tensor. The mean is calculated using the torch.mean function along
    the second dimension, unsqueezed to add a new dimension, and repeated along the second dimension
    using the repeat method of the tensor. The resulting tensor is then returned as the output of 
    the forward method.

    Note that this implementation assumes that the input tensor represents the seasonal part of a 
    time series data. The purpose of subtracting the mean is to ensure that the output has a zero 
    mean in each channel.
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series

    kernel_size: The size of the kernel to be used in the 1D average pooling operation.
    stride: The stride of the pooling operation.

    Input tensor x with the shape (batch_size, sequence_length, input_size)

    The forward function takes an input tensor x of shape (batch_size, sequence_length, input_size) 
    and applies a moving average operation on the time dimension (dimension 1) of the input. 
    The implementation first pads the input tensor at both ends with a reflection of the first/last 
    element in the time dimension, and then applies 1D average pooling with the specified kernel size 
    and stride. Finally, the output tensor is returned with the same shape as the input tensor.
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size,
                                stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(
            1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1,
                                  math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block

    The constructor of this class takes an integer kernel_size as input, which represents the kernel size 
    of the moving average filter. In the __init__ method, it initializes a moving_avg instance with 
    kernel_size as the input kernel size and stride=1.

    The forward method takes an input tensor x of shape (batch_size, sequence_length, input_size) and applies 
    the moving_avg instance to the input tensor. The resulting tensor is subtracted from the input tensor to 
    obtain the residual tensor. The residual tensor and the moving average tensor are returned as the output
    of the forward method.

    Note that this implementation assumes that the input tensor has a time-domain signal along the second dimension,
    and the output tensors will have the same shape as the input tensor. The purpose of this block is to decompose a
    time series signal into a trend and a seasonal component. The moving average filter is used to extract the seasonal
    component of the signal, which is then subtracted from the input signal to obtain the trend component.
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block

    The constructor of this class takes a list of kernel sizes kernel_size as input. 
    In the __init__ method, it initializes multiple moving_avg instances with each kernel 
    size, and a Linear layer with input size 1 and output size equal to the number of kernel sizes.

    The forward method takes an input tensor x of shape (batch_size, sequence_length, input_size) and 
    first applies each moving_avg instance to x. The resulting tensors are concatenated along the last
    dimension and passed through the Linear layer followed by a softmax function to obtain a set of weights 
    for each kernel size. The weighted average of the moving averages is then computed, and the resulting
    tensor is subtracted from x to obtain the residual tensor. The residual tensor and the weighted average
    tensor are returned as the output of the forward method.

    Note that this implementation assumes that the input tensor has a time-domain signal along the second 
    dimension, and the output tensors will have the same shape as the input tensor.
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1)
                           for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(
            moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


class FourierDecomp(nn.Module):
    """
    The constructor of this class does not define any parameters. In the forward function, 
    it takes an input tensor x of shape (batch_size, ..., sequence_length) and applies the 
    real-valued fast Fourier transform (FFT) to the last dimension (i.e., dim=-1) of the 
    input tensor. The resulting tensor x_ft will have shape (batch_size, ..., sequence_length//2+1, 2) 
    where the last dimension represents the real and imaginary parts of the Fourier coefficients.
    
    """
    def __init__(self):
        super(FourierDecomp, self).__init__()
        pass

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture

    The EncoderLayer class takes as input a multi-head self-attention module (defined outside of this class),
    the dimension of the model (d_model), the number of neurons in the feedforward sublayer (d_ff), a kernel 
    size (moving_avg) for a series decomposition block, the dropout probability, and an activation function 
    (either ReLU or GeLU).

    The forward() method applies the self-attention module to the input tensor x, then adds a residual connection 
    with dropout. It then applies a series decomposition block to the output of this step. The result of this step 
    is stored in x and y, with x being passed through a convolutional feedforward sublayer, and then added to y 
    before passing through another series decomposition block. The final output of the forward() method is the 
    residual output of this second decomposition block, as well as the attention tensor returned by the attention 
    module.
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(
            conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """

    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
            self.decomp3 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
            self.decomp3 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])

        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(
            residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
