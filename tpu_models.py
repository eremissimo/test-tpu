import torch
import torch.nn as nn
import torch.nn.functional as ff
from padding import get_padding


def t3(val):
    return val, val, val


class ResLayer(nn.Module):
    """ ResNet IDLayer for shallow query, key, value nets """
    def __init__(self, in_channels, out_channels=None, kernel_size=3, resampling=0, use_norm=True):
        super().__init__()
        mid_channels = out_channels if out_channels is not None else int(in_channels / (2**resampling))
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
            nn.Conv3d(n_chan, 4, kernel_size=t3(3), padding=1, bias=True)
        )


class SImpleUnet(nn.Module):
    def __init__(self, n_chan: int, kernel_size: int = 3, use_norm: bool = True, skip_conn_op: str = 'cat'):
        super().__init__()
        self.proj_in = nn.Sequential(
            nn.Conv3d(4, n_chan, kernel_size=t3(kernel_size), padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        if skip_conn_op == 'cat':
            self.skip_conn_func = torch.add
            skip_chan_mult = 1
        elif skip_conn_op == 'sum':
            self.skip_conn_func = lambda x, y: torch.cat((x, y), dim=1)
            skip_chan_mult = 2
        else:
            raise ValueError("Just use skip_conn_op = 'cat' or 'sum'")
        self.d0 = ResLayer(n_chan, resampling=0, kernel_size=kernel_size, use_norm=use_norm)
        self.d2 = ResLayer(n_chan, resampling=-1, kernel_size=kernel_size, use_norm=use_norm)
        self.d4 = ResLayer(2*n_chan, resampling=-1, kernel_size=kernel_size, use_norm=use_norm)
        self.bottleneck = ResLayer(4*n_chan, resampling=0, kernel_size=kernel_size, use_norm=use_norm)
        self.u4 = ResLayer(skip_chan_mult*4*n_chan, 2*n_chan, kernel_size=kernel_size, resampling=1, use_norm=use_norm)
        self.u2 = ResLayer(skip_chan_mult*2*n_chan, n_chan, kernel_size=kernel_size, resampling=1, use_norm=use_norm)
        self.u0 = ResLayer(skip_chan_mult*n_chan, n_chan, kernel_size=kernel_size, resampling=0, use_norm=use_norm)
        self.proj_out = nn.Conv3d(n_chan, 4, kernel_size=t3(kernel_size), padding=kernel_size//2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.d0(self.proj_in(x))
        x2 = self.d2(x0)
        x4 = self.d4(x2)
        x4 = self.skip_conn_func(x4, self.bottleneck(x4))
        x2 = self.skip_conn_func(x2, self.u4(x4))
        x0 = self.skip_conn_func(x0, self.u2(x2))
        x0 = self.proj_out(self.u0(x0))
        return x0


if __name__ == "__main__":
    tmp_in = torch.randn((2, 4, 64, 64, 32))
    tmp_out = SImple(8)(tmp_in)
    print(tmp_in.shape)
    print(tmp_out.shape)

    tmp_out = SImpleUnet(8)(tmp_in)
    print(tmp_in.shape)
    print(tmp_out.shape)

    print("woah!")
