import torch
import torch.nn as nn
import torch.nn.functional as ff
from padding import get_padding


def t3(val):
    return val, val, val


class Resample(nn.Module):
    def __init__(self, scale_factor=1):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return x if (self.scale_factor == 1) else ff.interpolate(x, scale_factor=self.scale_factor)


class ResLayer(nn.Module):
    """ ResNet IDLayer """
    def __init__(self, in_channels: int, kernel_size: int = 3, resampling: int = 0, use_norm: bool =True):
        super().__init__()
        mid_channels = int(in_channels / (2**resampling))
        stride = 1 if resampling == 0 else 2
        padd = t3(get_padding(kernel_size, stride=stride, dilation=1))
        # for convtranspose:
        ct_kernel_size = kernel_size + 1
        ct_padding = t3(get_padding(ct_kernel_size, stride=stride, dilation=1) - 1)
        # conv or conv-transpose
        if resampling <= 0:
            resample_layer = nn.Conv3d(in_channels, mid_channels, kernel_size=t3(kernel_size), padding=padd,
                                       stride=t3(stride), bias=False)
        else:
            resample_layer = nn.ConvTranspose3d(in_channels, mid_channels, kernel_size=t3(ct_kernel_size),
                                                padding=ct_padding, stride=t3(2), bias=False)
        module_list = []
        if use_norm:
            module_list.append(nn.BatchNorm3d(in_channels))
        module_list.append(nn.Conv3d(in_channels, in_channels, kernel_size=t3(kernel_size),
                                     padding=(kernel_size//2), bias=False))
        module_list.append(nn.ReLU(inplace=True))
        if use_norm:
            module_list.append(nn.BatchNorm3d(in_channels))
        module_list.append(resample_layer)
        self.main_branch = nn.Sequential(*module_list)

        if resampling < 0:
            self.skip_branch = nn.Conv3d(in_channels, mid_channels, kernel_size=t3(1), stride=t3(2), bias=False)
        elif resampling == 0:
            self.skip_branch = nn.Identity()
        else:
            self.skip_branch = nn.ConvTranspose3d(in_channels, mid_channels, kernel_size=t3(2), stride=t3(2),
                                                  bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = self.main_branch(x) + self.skip_branch(x)
        op = self.relu(op)
        return op


class SImple(nn.Sequential):
    def __init__(self, n_chan, use_norm=True):
        super().__init__(
            nn.Conv3d(4, n_chan, kernel_size=t3(3), padding=1, bias=True),
            nn.ReLU(inplace=True),
            ResLayer(n_chan, resampling=0, use_norm=use_norm),
            ResLayer(n_chan, resampling=-1, use_norm=use_norm),
            ResLayer(2*n_chan, resampling=-1, use_norm=use_norm),
            ResLayer(4*n_chan, resampling=0, use_norm=use_norm),
            ResLayer(4*n_chan, resampling=1, use_norm=use_norm),
            ResLayer(2*n_chan, resampling=1, use_norm=use_norm),
            ResLayer(n_chan, resampling=0, use_norm=use_norm),
            nn.Conv3d(n_chan, 4, kernel_size=t3(3), padding=1, bias=False)
        )


"""    #############################    """


class ContrastAdjustment(nn.Module):
    """ A simple elementwise remapping of the input featuremap x = f(x), where f is parametrized as a sum of the
        identity and gaussian terms with learnable parameters
        f(x) = x + sum(a[i] * N(x | m[i], s[i]))

        On init the module initializes as near-identity
    """
    def __init__(self, n_chan: int,  num_terms: int = 2):
        super().__init__()
        # [terms, B, chan, X, Y, Z] broadcasting scheme
        sz = (num_terms, 1, n_chan, 1, 1, 1)
        self.amplitudes = nn.Parameter(0.032*torch.randn(sz), requires_grad=True)
        self.means = nn.Parameter(torch.rand(sz), requires_grad=True)
        self.stds = nn.Parameter(0.25*torch.ones(sz, dtype=torch.float32), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # scaling input tensor to [0,1]
        mx = torch.max(x)
        mn = torch.min(x)
        op = (x - mn)/(mx - mn)
        # applying transform
        op = op + (self.amplitudes*torch.exp(-0.5*((op - self.means)/self.stds).square())).sum(dim=0)
        # rescaling back
        op = mn + op*(mx - mn)
        return op


class ContrNet(nn.Sequential):
    def __init__(self, n_chan: int, kernel_size=3, use_norm: bool = True):
        super().__init__(
            ContrastAdjustment(4),
            nn.Conv3d(4, n_chan, kernel_size=t3(3), padding=1, bias=True),
            nn.ReLU(inplace=True),
            ResLayer(n_chan, resampling=0, kernel_size=kernel_size, use_norm=use_norm),
            ContrastAdjustment(n_chan),
            ResLayer(n_chan, resampling=0, kernel_size=kernel_size, use_norm=use_norm),
            ContrastAdjustment(n_chan),
            ResLayer(n_chan, resampling=0, kernel_size=kernel_size, use_norm=use_norm),
            ContrastAdjustment(n_chan),
            ResLayer(n_chan, resampling=0, kernel_size=kernel_size, use_norm=use_norm),
            ContrastAdjustment(n_chan),
            ResLayer(n_chan, resampling=0, kernel_size=kernel_size, use_norm=use_norm),
            nn.Conv3d(n_chan, 4, kernel_size=t3(3), padding=1, bias=False)
        )


class MaskedNet(nn.Module):
    """ A simple mask application to intermediate featuremap of the main_net.
     Mask_net and main_net[connect_idx-1], both must have the same output shape. """
    def __init__(self, main_net: nn.Sequential, mask_net: nn.Module, connect_idx: int):
        super().__init__()
        self.mask_net = mask_net
        self.m1net = main_net[:connect_idx]
        self.m2net = main_net[connect_idx:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = self.m1net(x)*self.mask_net(x)     # this!
        op = self.m2net(op)
        return op


if __name__ == "__main__":
    tmp_in = torch.randn((1, 4, 64, 64, 32))
    tmp_out = SImple(8)(tmp_in)
    print(tmp_in.shape)
    print(tmp_out.shape)

    tmp_in = tmp_in.cuda()
    model = ContrNet(8, kernel_size=5).cuda()
    tmp_out = model(tmp_in)
    print(tmp_in.shape)
    print(tmp_out.shape)

    print("woah!")
