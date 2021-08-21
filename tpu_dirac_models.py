""" From https://github.com/szagoruyko/diracnets.
    DiracNets: Training Very Deep Neural Networks Without Skip-Connections
    https://arxiv.org/abs/1706.00388
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import dirac_
from padding import get_padding


def t3(val):
    return val, val, val


def normalize(w):
    """Normalizes weight tensor over full filter."""
    return F.normalize(w.view(w.shape[0], -1)).view_as(w)


class DiracConv(nn.Module):
    def init_params(self, out_channels):
        self.alpha = nn.Parameter(torch.Tensor(out_channels).fill_(1))
        self.beta = nn.Parameter(torch.Tensor(out_channels).fill_(0.1))
        self.register_buffer('delta', dirac_(self.weight.data.clone()))
        assert self.delta.shape == self.weight.shape
        self.v = (-1,) + (1,) * (self.weight.dim() - 1)

    def transform_weight(self):
        return self.alpha.view(*self.v) * self.delta + self.beta.view(*self.v) * normalize(self.weight)


class DiracConv3d(nn.Conv3d, DiracConv):
    """Dirac parametrized convolutional layer.
    Works the same way as `nn.Conv3d`, but has additional weight parametrizatoin:
        :math:`\alpha\delta + \beta W`,
    where:
        :math:`\alpha` and :math:`\beta` are learnable scalars,
        :math:`\delta` is such a tensor so that `F.conv3d(x, delta) = x`, ie
            Kroneker delta
        `W` is weight tensor
    It is user's responsibility to set correcting padding. Only stride=1 supported.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias)
        self.init_params(out_channels)

    def forward(self, input):
        return F.conv3d(input, self.transform_weight(), self.bias, self.stride, self.padding, self.dilation)


class SImpleDirac(nn.Sequential):
    """DiracConv equivalent of SImple encoder-decoder segmentation model"""
    def __init__(self, n_chan):
        super().__init__(
            DiracConv3d(4, n_chan, kernel_size=t3(3), padding=1, bias=False),
            nn.ReLU(),
            DiracConv3d(n_chan, n_chan, kernel_size=t3(3), padding=1, bias=False),
            nn.MaxPool3d(2, 2),
            nn.ReLU(),
            DiracConv3d(n_chan, 2*n_chan, kernel_size=t3(3), padding=1, bias=False),
            nn.MaxPool3d(2, 2),
            nn.ReLU(),
            DiracConv3d(2*n_chan, 4*n_chan, kernel_size=t3(3), padding=1, bias=False),
            nn.ReLU(),
            DiracConv3d(4*n_chan, 4*n_chan, kernel_size=t3(3), padding=1, bias=False),
            nn.ReLU(),
            DiracConv3d(4*n_chan, 2*n_chan, kernel_size=t3(3), padding=1, bias=False),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            DiracConv3d(2*n_chan, n_chan, kernel_size=t3(3), padding=1, bias=False),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            DiracConv3d(n_chan, n_chan, kernel_size=t3(3), padding=1, bias=False),
            nn.ReLU(),
            nn.Conv3d(n_chan, 4, kernel_size=t3(3), padding=1, bias=False)
        )


if __name__ == "__main__":
    tmp_in = torch.randn((2, 4, 64, 64, 32))
    tmp_out = SImpleDirac(8)(tmp_in)
    print(tmp_in.shape)
    print(tmp_out.shape)

    print("woah!")