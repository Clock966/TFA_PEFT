#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional
from model.feature import IPDFeature, ConvSTFT, ConviSTFT
from torch import Tensor


class caft(nn.Module):
    def __init__(self, channel=16, reduction=32):
        super(caft, self).__init__()
        if channel // 32 == 0:
            reduction = 1
        self.conv_f = nn.Conv1d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                bias=False)
        self.conv_t = nn.Conv1d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                bias=False)

        self.relu = nn.ReLU()

        self.att_f = nn.Conv1d(in_channels=channel // reduction,out_channels=channel // reduction,kernel_size=3,stride=1,padding=1)
        self.att_t = nn.Conv1d(in_channels=channel // reduction,out_channels=channel // reduction,kernel_size=7,stride=1,padding=3)


        #self.bn = nn.GroupNorm(num_groups=channel,num_channels=channel)
        self.F_f = nn.Conv1d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_t = nn.Conv1d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.sigmoid_f = nn.Sigmoid()
        self.sigmoid_t = nn.Sigmoid()

    def forward(self, x):
        b, c, f, t = x.shape
        avg_pool_f = nn.AdaptiveAvgPool2d((f, 1))
        avg_pool_t = nn.AdaptiveAvgPool2d((1, t))

        x_f = avg_pool_f(x).permute(0, 1, 3, 2).squeeze(-2)
        x_t = avg_pool_t(x).squeeze(-2)

        x_f_conv_relu = self.att_f(self.relu(self.conv_f(x_f)))
        x_t_conv_relu = self.att_t(self.relu(self.conv_t(x_t)))

        s_f = self.sigmoid_f(self.F_f(x_f_conv_relu).unsqueeze(-2).permute(0, 1, 3, 2))
        s_t = self.sigmoid_t(self.F_t(x_t_conv_relu)).unsqueeze(-2)

        out = x * s_f.expand_as(x) * s_t.expand_as(x)
        return out

class scCAFT(nn.Module):
    def __init__(self,channel=16):
        super(scCAFT, self).__init__()
        self.realcaft = caft(channel=channel)
        self.imagcaft = caft(channel=channel)

    def forward(self, x):
        xr, xi = torch.chunk(x, 2, -2)
        yr = self.realcaft(xr)
        yi = self.imagcaft(xi)
        y = torch.cat([yr, yi], -2)
        return y


class AdapterBlock(nn.Module):
    def __init__(self, in_dim, stride=1, bias=False):
        super(AdapterBlock, self).__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=stride, bias=bias, groups=in_dim, padding='same')
        self.relu1 = nn.ReLU(inplace=True)
        # self.se1 = SELayer(out_dim)
        self.conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=5, stride=stride, bias=False, groups=in_dim, padding='same')
        # self.se2 = SELayer(out_dim)
        self.conv3 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=stride, bias=bias, groups=in_dim, padding='same')
        # self.relu2 = nn.ReLU(inplace=True)
        self.sccaft = scCAFT(in_dim)

    # self.layer_norm2 = nn.LayerNorm(out_dim)
    # self.dropout = nn.Dropout(p=0.1)
    def forward(self, x, residual_input):

        out = self.ln(x.transpose(1, 3)).transpose(1, 3)
        #out = torch.transpose(out, -1, -2)
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.sccaft(out)
        out = residual_input + out  # skip connection
        return out

def parse_1dstr(sstr: str) -> List[int]:           # 渚�: "1,1,1,1,1" -> [1, 1, 1, 1, 1]
    return list(map(int, sstr.split(",")))


def parse_2dstr(sstr: str) -> List[List[int]]:     # 渚�: "3,3;3,3;3,3;3,3;3,3" -> [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    return [parse_1dstr(tok) for tok in sstr.split(";")]


class ComplexConv2d(nn.Module):
    """
    Complex 2D Convolution
    """

    def __init__(self, *args, **kwargs):
        super(ComplexConv2d, self).__init__()
        self.real = nn.Conv2d(*args, **kwargs)
        self.imag = nn.Conv2d(*args, **kwargs)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (Tensor): N x C x 2F x T
        Return:
            y (Tensor): N x C' x 2F' x T'
        """
        xr, xi = th.chunk(x, 2, -2)
        yr = self.real(xr) - self.imag(xi)
        yi = self.imag(xr) + self.real(xi)
        y = th.cat([yr, yi], -2)
        return y


class ComplexConvTranspose2d(nn.Module):
    """
    Complex Transpose 2D Convolution
    """

    def __init__(self, *args, **kwargs):
        super(ComplexConvTranspose2d, self).__init__()
        self.real = nn.ConvTranspose2d(*args, **kwargs)
        self.imag = nn.ConvTranspose2d(*args, **kwargs)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (Tensor): N x C x 2F x T
        Return:
            y (Tensor): N x C' x 2F' x T'
        """
        xr, xi = th.chunk(x, 2, -2)
        yr = self.real(xr) - self.imag(xi)
        yi = self.imag(xr) + self.real(xi)
        y = th.cat([yr, yi], -2)
        return y


class ComplexBatchNorm2d(nn.Module):
    """
    A easy implementation of complex 2d batchnorm
    """

    def __init__(self, *args, **kwargs):
        super(ComplexBatchNorm2d, self).__init__()
        self.real_bn = nn.BatchNorm2d(*args, **kwargs)
        self.imag_bn = nn.BatchNorm2d(*args, **kwargs)

    def forward(self, x: th.Tensor) -> th.Tensor:
        xr, xi = th.chunk(x, 2, -2)
        xr = self.real_bn(xr)
        xi = self.imag_bn(xi)
        x = th.cat([xr, xi], -2)
        return x


class EncoderBlock(nn.Module):
    """
    Convolutional block in encoder
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int],
                 stride: int = 1,
                 padding: int = 0,
                 causal: bool = False,
                 cplx: bool = True) -> None:
        super(EncoderBlock, self).__init__()

        conv_impl = ComplexConv2d if cplx else nn.Conv2d

        # NOTE: time stride should be 1
        var_kt = kernel_size[1] - 1

        time_axis_pad = var_kt if causal else var_kt // 2

        self.conv = conv_impl(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=(padding, time_axis_pad))
        self.adapter = AdapterBlock(in_dim=in_channels)
        if cplx:
            self.bn = ComplexBatchNorm2d(out_channels)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.causal = causal
        self.time_axis_pad = time_axis_pad

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (Tensor): N x 2C x F x T
        """
        b,c,f,t = x.shape
        if c !=1:
            x = self.adapter(x, x)
        #x = self.adapter(x, x)

        x = self.conv(x)

        if self.causal:
            x = x[..., :-self.time_axis_pad]

        x = self.bn(x)
        x = F.leaky_relu(x)
        return x


class DecoderBlock(nn.Module):
    """
    Convolutional block in decoder
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int],
                 stride: int = 1,
                 padding: int = 0,
                 output_padding: int = 0,
                 causal: bool = False,
                 cplx: bool = True,
                 last_layer: bool = False) -> None:
        super(DecoderBlock, self).__init__()
        conv_impl = ComplexConvTranspose2d if cplx else nn.ConvTranspose2d
        var_kt = kernel_size[1] - 1
        time_axis_pad = var_kt if causal else var_kt // 2
        self.trans_conv = conv_impl(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride=stride,
                                    padding=(padding, var_kt - time_axis_pad),
                                    output_padding=(output_padding, 0))
        if last_layer:
            self.bn = None
        else:
            if cplx:
                self.bn = ComplexBatchNorm2d(out_channels)
            else:
                self.bn = nn.BatchNorm2d(out_channels)
        self.causal = causal
        self.time_axis_pad = time_axis_pad
        self.adapter = AdapterBlock(in_dim=in_channels)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (Tensor): N x 2C x F x T
        """
        x = self.adapter(x, x)
        x = self.trans_conv(x)
        if self.causal:
            x = x[..., :-self.time_axis_pad]
        if self.bn:
            x = self.bn(x)
            x = F.leaky_relu(x)
        return x
class Encoder(nn.Module):
    """
    Encoder of the UNet
        K: filters
        S: strides
        C: output channels
    """

    def __init__(self,
                 cplx: bool,
                 K: List[Tuple[int, int]],
                 S: List[Tuple[int, int]],
                 C: List[int],
                 P: List[int],
                 causal: bool = False) -> None:
        super(Encoder, self).__init__()
        layers = [
            EncoderBlock(C[i],
                         C[i + 1],
                         k,
                         stride=S[i],
                         padding=P[i],
                         cplx=cplx,
                         causal=causal) for i, k in enumerate(K)
        ]
        self.layers = nn.ModuleList(layers)
        self.num_layers = len(layers)
    def forward(self, x: th.Tensor) -> Tuple[List[th.Tensor], th.Tensor]:
        enc_h = []
        for index, layer in enumerate(self.layers):
            x = layer(x)
            # print(f"encoder-{index}: {x.shape}")
            if index + 1 != self.num_layers:

                enc_h.append(x)
        return enc_h, x


class Decoder(nn.Module):
    """
    Decoder of the UNet
        K: filters
        S: strides
        C: output channels
    """

    def __init__(self,
                 cplx: bool,
                 K: List[Tuple[int, int]],
                 S: List[Tuple[int, int]],
                 C: List[int],
                 P: List[int],
                 O: List[int],
                 causal: bool = False,
                 connection: str = "sum") -> None:
        super(Decoder, self).__init__()
        if connection not in ["cat", "sum"]:
            raise ValueError(f"Unknown connection mode: {connection}")
        layers = [
            DecoderBlock(C[i] * 2 if connection == "cat" and i != 0 else C[i],
                         C[i + 1],
                         k,
                         stride=S[i],
                         padding=P[i],
                         output_padding=O[i],
                         causal=causal,
                         cplx=cplx,
                         last_layer=(i == len(K) - 1)) for i, k in enumerate(K)
        ]
        self.layers = nn.ModuleList(layers)
        self.connection = connection

    def forward(self, x: th.Tensor, enc_h: List[th.Tensor]) -> th.Tensor:
        # N = len(self.layers)
        for index, layer in enumerate(self.layers):
            if index == 0:
                x = layer(x)
            else:
                # N x C x F x T
                if self.connection == "sum":
                    inp = x + enc_h[index - 1]
                else:
                    # N x 2C x F x T
                    inp = th.cat([x, enc_h[index - 1]], 1)
                x = layer(inp)
            # print(f"decoder-{N - 1 - index}: {x.shape}")
        return x


# @ApsRegisters.sse.register("sse@dcunet")
class CAFTUNet(nn.Module):
    """
    Real or Complex UNet for Speech Enhancement

    Args:
        K, S, C: kernel, stride, channel size for convolution in encoder/decoder
        P: padding on frequency axis for convolution in encoder/decoder
        O: output_padding on frequency axis for transposed_conv2d in decoder
    NOTE: make sure that stride size on time axis is 1 (we do not do subsampling on time axis)
    """

    def __init__(self,
                 cplx: bool = True,        # whether use complex
                 K: str = "3,3;3,3;3,3;3,3;3,3;3,3;3,3;3,3",
                 S: str = "2,1;2,1;2,1;2,1;2,1;2,1;2,1;2,1",
                 #C: str = "32,32,64,64,64,64,128,128",
                 C: str = "32,32,64,64,64,64,128,128",
                 P: str = "1,1,1,1,1,1,1,1",
                 O: str = "0,0,0,0,0,0,0,0",
                 #K: str = "3,5;3,5;3,5;3,5;3,5",

                 #K: str = "3,5;3,5;3,5;3,5;3,3",
                 #K: str = "7,5;7,5;5,3;5,3;5,3",
                 #O: str = "0,0,1,1,1",
                 #K: str = "7,5;5,3;5,3;5,3;5,3",
                 #K: str = "7,5;5,3;5,3;5,3;5,3",
                 # K: str = "3,3;3,3;3,3;3,3;3,3",
                 # O: str = "0,0,0,0,0",
                 # S: str = "2,1;2,1;2,1;2,1;2,1",
                 # C: str = "16,32,32,64,64",
                 # P: str = "1,1,1,1,1",


                 #K: str = "7,5;7,5;7,5;5,3;5,3;5,3;5,3",
                 #S: str = "2,1;2,1;2,1;2,1;2,1;2,1;2,1",
                 #C: str = "32,32,64,64,64,64,64",
                 #P: str = "1,1,1,1,1,1,1",
                 #O: str = "0,0,0,0,0,0,0",

                 num_branch: int = 1,
                 causal_conv: bool = False,
                 win_len=512,
                 win_inc=256,
                 fft_len=512,
                 win_type='hamming',  # 濡傛灉鏀硅緭鍏ラ€氶亾鏁板垯闇€瑕佹敼涓嬮潰涓や釜鍙傛暟
                 mix_channels=1,      # nearend_mic_signal channels
                 #far_channels=1,      # farend_speech channels
                 #enh_transform: Optional[nn.Module] = None,
                 freq_padding: bool = True,
                 connection: str = "cat") -> None:
        #super(DCUNet, self).__init__(enh_transform, training_mode="freq")
        super(CAFTUNet, self).__init__()
        #assert enh_transform is not None
        #self.normalize=1
        #self.floor=0.001
        self.cplx = cplx
        self.mix_channels = mix_channels
        #self.bn1d = th.nn.BatchNorm1d(out_channels+2)
        #self.forward_stft = enh_transform.ctx(name="forward_stft")
        #self.inverse_stft = enh_transform.ctx(name="inverse_stft")
        self.stft = ConvSTFT(win_len=win_len,
                             win_inc=win_inc,
                             fft_len=fft_len,
                             win_type=win_type,
                             )

        self.istft = ConviSTFT(win_len=win_len,
                               win_inc=win_inc,
                               fft_len=fft_len,
                               win_type=win_type,
                               )
        K = parse_2dstr(K)
        S = parse_2dstr(S)
        C = parse_1dstr(C)
        P = parse_1dstr(P)
        O = parse_1dstr(O)

        self.encoder = Encoder(cplx, K, S, [mix_channels] + C, P, causal=causal_conv)


        self.decoder = Decoder(cplx,
                               K[::-1],       # reverse
                               S[::-1],
                               C[::-1] + [mix_channels],
                               P[::-1],
                               O[::-1],
                               causal=causal_conv,
                               connection=connection)
        self.num_branch = num_branch
    '''
    def sep(self, m: th.Tensor, sr: th.Tensor, si: th.Tensor) -> th.Tensor:
        # m: N x 2F x T
        if self.cplx:
            # N x F x T
            mr, mi = th.chunk(m, 2, -2)
            m_abs = (mr**2 + mi**2)**0.5
            m_mag = th.tanh(m_abs)
            mr, mi = m_mag * mr / m_abs, m_mag * mi / m_abs
            s = self.inverse_stft(sr * mr - si * mi, sr * mi + si * mr, cplx=True)
        else:
            s = self.inverse_stft(sr * m, si * m, cplx=True)
        return s

    def infer(self,
              mix: th.Tensor,
              mode="time") -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): S
        Return:
            Tensor: S
        """
        self.check_args(mix, training=False, valid_dim=[1])
        with th.no_grad():
            mix = mix[None, :]
            sep = self.forward(mix)
            if self.num_branch == 1:
                return sep[0]
            else:
                return [s[0] for s in sep]
    '''
    #def forward(self, s: th.Tensor) -> Union[th.Tensor, List[th.Tensor]]:
    #def forward(self, mix: th.Tensor, echo:th.Tensor):
    def forward(self, mix: th.Tensor):
        yr, yi = self.stft(mix, cplx=True)

        if self.cplx:
            # N x C x 2F x T
            inp = (th.cat([yr, yi], -2)).contiguous()    # 姝ゅinp涓簃ix鐨勫疄閮ㄨ櫄閮ㄦ嫾鎺�
        else:
            # N x C x F x T
            inp = ((yr**2 + yi**2)**0.5).contiguous()    # 姝ゅinp涓簃ix鐨勫箙搴﹁氨

        # encoder
        #enc_h, h = self.encoder(s[:, None])
        enc_h, h = self.encoder(inp)
        # reverse
        enc_h = enc_h[::-1]
        # decoder
        m = self.decoder(h, enc_h)


        if  self.cplx:
        # N x C x 2F x T
            mr, mi = th.chunk(m, 2, -2)
            zr = yr * mr - yi * mi         # cIRM
            zi = yr * mi + yi * mr
        else:
            zr = yr*m
            zi = yi*m

        #return zr, zi, m

        N, C, F, T = yr.shape
        #zr0 = zr
        #zi0 = zi
        zr = zr.reshape(N * C, zr.shape[-2], -1)
        zi = zi.reshape(N * C, zi.shape[-2], -1)
        out_wav = self.istft(zr, zi, cplx=True)
        # out_wav = PCSwav(zr, zi)
        out_wav = out_wav.view(N, C, out_wav.shape[-1])
        out_wav = th.clamp(out_wav, -1, 1)
        # out_wav =out_wav[:, 0:1, :]
        '''
        est = {
            "wav": out_wav,
        }
        '''
        # est = out_wav
        return out_wav

if __name__ == "__main__":
    from thop import profile, clever_format
    mix = th.randn(1, 1, 16000)
    net = CAFTUNet()
    est = net(mix)
    print(net)
    print(est.shape)
    macs, params = profile(net, inputs=(mix,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('flops: ', macs)
    print('params: ', params)
    print("Total params: %.2fM" % (sum(p.numel() for p in net.parameters()) / 1e6))

    # for name, param in net.named_parameters():
    #     if 'adapter' not in name:
    #         param.requires_grad = False
    # # for name, param in net.named_parameters():
    #     if not param.requires_grad:
    #         print(name,"False")
    #     else:
    #         print(name, "true")