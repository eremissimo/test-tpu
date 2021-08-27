import torch
import torch.nn as nn
from padding import get_padding
from typing import Tuple


"""    #############################    """
"""    #          Layers           #    """
"""    #############################    """


def t3(val):
    return val, val, val


def t2(val):
    return val, val


def t1(val):
    return val


class ResLayer(nn.Module):
    """ ResNet Layer """
    def __init__(self, in_channels: int, kernel_size: int = 3, resampling: int = 0, use_norm: bool = True,
                 ndim: int = 3):
        super().__init__()
        tn = [t1, t2, t3][ndim-1]
        Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][ndim-1]
        Convtr = [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d][ndim-1]
        Bn = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][ndim-1]
        mid_channels = int(in_channels / (2**resampling))
        stride = 1 if resampling == 0 else 2
        padd = tn(get_padding(kernel_size, stride=stride, dilation=1))
        # for convtranspose:
        ct_kernel_size = kernel_size + 1
        ct_padding = tn(get_padding(ct_kernel_size, stride=stride, dilation=1) - 1)
        # conv or conv-transpose
        if resampling <= 0:
            resample_layer = Conv(in_channels, mid_channels, kernel_size=tn(kernel_size), padding=padd,
                                  stride=tn(stride), bias=False)
        else:
            resample_layer = Convtr(in_channels, mid_channels, kernel_size=tn(ct_kernel_size),
                                    padding=ct_padding, stride=tn(2), bias=False)
        module_list = []
        if use_norm:
            module_list.append(Bn(in_channels))
        module_list.append(Conv(in_channels, in_channels, kernel_size=tn(kernel_size),
                                padding=(kernel_size//2), bias=False))
        module_list.append(nn.ReLU(inplace=True))
        if use_norm:
            module_list.append(Bn(in_channels))
        module_list.append(resample_layer)
        self.main_branch = nn.Sequential(*module_list)

        if resampling < 0:
            self.skip_branch = Conv(in_channels, mid_channels, kernel_size=tn(1), stride=tn(2), bias=False)
        elif resampling == 0:
            self.skip_branch = nn.Identity()
        else:
            self.skip_branch = Convtr(in_channels, mid_channels, kernel_size=tn(2), stride=tn(2),
                                      bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = self.main_branch(x) + self.skip_branch(x)
        op = self.relu(op)
        return op


class ResLayer1d(ResLayer):
    def __init__(self, in_channels: int, kernel_size: int = 3, resampling: int = 0, use_norm: bool = True):
        super().__init__(in_channels, kernel_size, resampling, use_norm, ndim=1)


class ResLayer2d(ResLayer):
    def __init__(self, in_channels: int, kernel_size: int = 3, resampling: int = 0, use_norm: bool = True):
        super().__init__(in_channels, kernel_size, resampling, use_norm, ndim=2)


class ResLayer3d(ResLayer):
    def __init__(self, in_channels: int, kernel_size: int = 3, resampling: int = 0, use_norm: bool = True):
        super().__init__(in_channels, kernel_size, resampling, use_norm, ndim=3)


class To2d(nn.Module):
    """ Reshaping 3d image to 2d by flattening one chosen spatial dimension with the batch dimension """
    def __init__(self, dim_to_batch: int = 2):
        super().__init__()
        spatials_dim_idxs = [2, 3, 4]
        del spatials_dim_idxs[dim_to_batch - 2]
        self.perm_dims = [0, dim_to_batch, 1, *spatials_dim_idxs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dims: batch, channels, *spatial dims
        # [B, C, X, Y, Z] -> [B, X, C, Y, Z] -> [B*X, C, Y, Z]
        # op = x.transpose(1, 2).flatten(start_dim=0, end_dim=1)
        op = x.permute(self.perm_dims).flatten(start_dim=0, end_dim=1)   # maybe .contiguous() ?
        return op


class To3d(nn.Module):
    """ Inverse of To2d """
    def __init__(self, original_batch_size: int, dim_from_batch: int = 2):
        super().__init__()
        self.batch_size = original_batch_size
        spatials_dim_idxs = [3, 4]
        spatials_dim_idxs.insert(dim_from_batch - 2, 1)
        self.perm_dims = [0, 2, *spatials_dim_idxs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # restoring original dimensions
        # [B*X, C, Y, Z] -> [B, X, C, Y, Z] -> [B, C, X, Y, Z]
        # op = x.view(sz[0], sz[2], *x.shape[1:]).transpose(1, 2)
        op = x.view(self.batch_size, -1, *x.shape[1:]).permute(self.perm_dims)   # maybe .contiguous() ?
        return op


class ContrastAdjustment(nn.Module):
    """ A simple elementwise remapping of the input featuremap x = f(x), where f is parametrized as a sum of the
        identity and gaussian terms with learnable parameters
        f(x) = x*(1 + sum(a[i] * N(x | m[i], s[i])))

        On init the module initializes as near-identity
    """
    def __init__(self, n_chan: int,  num_terms: int = 2, nd: int = 3):
        super().__init__()
        # [terms, B, chan, X, Y, Z] broadcasting scheme
        shp = [num_terms, 1, n_chan] + [1]*nd
        self.amplitudes = nn.Parameter(0.032*torch.randn(shp), requires_grad=True)
        self.means = nn.Parameter(torch.rand(shp), requires_grad=True)
        self.stds = nn.Parameter(0.25*torch.ones(shp, dtype=torch.float32), requires_grad=True)

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


class ContrastAdjustment3d(ContrastAdjustment):
    def __init__(self, n_chan: int,  num_terms: int = 2):
        super().__init__(n_chan, num_terms, nd=3)


class ContrastAdjustment2d(ContrastAdjustment):
    def __init__(self, n_chan: int,  num_terms: int = 2):
        super().__init__(n_chan, num_terms, nd=2)


class ContrastAdjustment1d(ContrastAdjustment):
    def __init__(self, n_chan: int,  num_terms: int = 2):
        super().__init__(n_chan, num_terms, nd=1)


class DummyDebugLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"shape = {x.shape}")
        return x


"""    #############################    """


"""    #############################    """
"""    #          Baseline         #    """
"""    #############################    """


class SImple(nn.Sequential):
    def __init__(self, n_chan, use_norm=True):
        super().__init__(
            nn.Conv3d(4, n_chan, kernel_size=t3(3), padding=1, bias=True),
            nn.ReLU(inplace=True),
            ResLayer3d(n_chan, resampling=0, use_norm=use_norm),
            ResLayer3d(n_chan, resampling=-1, use_norm=use_norm),
            ResLayer3d(2*n_chan, resampling=-1, use_norm=use_norm),
            ResLayer3d(4*n_chan, resampling=0, use_norm=use_norm),
            ResLayer3d(4*n_chan, resampling=1, use_norm=use_norm),
            ResLayer3d(2*n_chan, resampling=1, use_norm=use_norm),
            ResLayer3d(n_chan, resampling=0, use_norm=use_norm),
            nn.Conv3d(n_chan, 4, kernel_size=t3(3), padding=1, bias=True)
        )


"""    #############################    """
"""    #          Conv232          #    """
"""    #############################    """


class Conv232d(nn.Sequential):
    def __init__(self, batch_size: int, encoder2d: nn.Module, bottleneck3d: nn.Module, decoder2d: nn.Module,
                 leaping_dim: int = 2):
        super().__init__(
            To2d(dim_to_batch=leaping_dim),
            encoder2d,
            To3d(batch_size, dim_from_batch=leaping_dim),
            bottleneck3d,
            To2d(dim_to_batch=leaping_dim),
            decoder2d,
            To3d(batch_size, dim_from_batch=leaping_dim)
        )


def _t31(k: int, i: int, defval: int = 1) -> Tuple[int, ...]:
    t = [defval, defval]
    t.insert(i, k)
    return tuple(t)


def conv232_assembly(n_chan: int, batch_size: int, use_norm: bool = True, leaping_dim: int = 2,
                     use_conadjust: bool = True):
    maybe_conadjust = ContrastAdjustment2d(4, 2) if use_conadjust else nn.Identity
    encoder2d = nn.Sequential(
        maybe_conadjust,
        nn.Conv2d(4, n_chan, kernel_size=t2(3), padding=1, bias=True),
        nn.ReLU(inplace=True),
        ResLayer2d(n_chan, resampling=0, use_norm=use_norm),
        ResLayer2d(n_chan, resampling=-1, use_norm=use_norm),
        ResLayer2d(2*n_chan, resampling=-1, use_norm=use_norm),
    )
    maybe_bn3d_1 = nn.BatchNorm3d(4*n_chan) if use_norm else nn.Identity()
    maybe_bn3d_2 = nn.BatchNorm3d(4*n_chan) if use_norm else nn.Identity()
    x_downsample = 4
    bottleneck3d = nn.Sequential(
        maybe_bn3d_1,
        nn.Conv3d(4*n_chan, 4*n_chan, kernel_size=_t31(x_downsample+1, leaping_dim-2),
                  stride=_t31(x_downsample, leaping_dim-2),
                  padding=_t31(x_downsample//2, leaping_dim-2, 0),
                  bias=False),
        nn.ReLU(inplace=True),
        ResLayer3d(4*n_chan, resampling=0, use_norm=use_norm),
        ResLayer3d(4*n_chan, resampling=0, use_norm=use_norm),
        ResLayer3d(4*n_chan, resampling=0, use_norm=use_norm),
        maybe_bn3d_2,
        nn.ConvTranspose3d(4*n_chan, 4*n_chan, kernel_size=_t31(x_downsample+1, leaping_dim-2),
                           stride=_t31(x_downsample, leaping_dim-2),
                           padding=_t31(x_downsample//2 - 1, leaping_dim-2, 0),
                           output_padding=_t31(1, leaping_dim-2, 0), bias=False),
        nn.ReLU(inplace=True)
    )
    decoder2d = nn.Sequential(
        ResLayer2d(4*n_chan, resampling=1, use_norm=use_norm),
        ResLayer2d(2*n_chan, resampling=1, use_norm=use_norm),
        ResLayer2d(n_chan, resampling=0, use_norm=use_norm),
        nn.Conv2d(n_chan, 4, kernel_size=t2(3), padding=1, bias=True)
    )
    model = Conv232d(batch_size, encoder2d, bottleneck3d, decoder2d, leaping_dim=leaping_dim)
    return model


"""    #############################    """


if __name__ == "__main__":
    tmp_in = torch.randn((2, 4, 128, 128, 80))
    tmp_out = SImple(8)(tmp_in)
    print(tmp_in.shape)
    print(tmp_out.shape)

    tmp_model = conv232_assembly(n_chan=3, batch_size=2)
    op2 = tmp_model(tmp_in)
    print(op2.shape)

    print("woah!")
