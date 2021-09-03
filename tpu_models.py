import torch
import torch.nn as nn
import torch.nn.functional as ff
from padding import get_padding
from typing import Tuple, Optional, List


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
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, kernel_size: int = 3, resampling: int = 0,
                 use_norm: bool = True, ndim: int = 3):
        super().__init__()
        tn = [t1, t2, t3][ndim-1]
        Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][ndim-1]
        Convtr = [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d][ndim-1]
        Bn = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][ndim-1]
        mid_channels = out_channels if out_channels is not None else int(in_channels / (2**resampling))
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
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, kernel_size: int = 3, resampling: int = 0,
                 use_norm: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, resampling, use_norm, ndim=1)


class ResLayer2d(ResLayer):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, kernel_size: int = 3, resampling: int = 0,
                 use_norm: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, resampling, use_norm, ndim=2)


class ResLayer3d(ResLayer):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, kernel_size: int = 3, resampling: int = 0,
                 use_norm: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, resampling, use_norm, ndim=3)


class UpsampleNd(nn.Module):
    """ Upsample as ConvTranspose with certain kernels. Interpolation is not supported by xla yet, that's why the
        native nn.Upsample is slow on TPU.
        Only natural scale factors are supported."""
    def __init__(self, in_channels: int, scale: int = 2, mode: str = "nearest", nd: int = 2):
        super().__init__()
        assert isinstance(scale, int)
        self.scale = scale
        if mode == "nearest":
            kernel_base1d = self._kernel_base_nearest(scale)
            self.padding = (2*scale - 1)//2
            self.outpadding = scale-1
        elif mode in {"linear", "bilinear", "trilinear"}:
            kernel_base1d = self._kernel_base_linear(scale)
            self.padding = scale // 2
            self.outpadding = 0
        else:
            raise ValueError(" Unexpected mode argument. Try using 'nearest' or 'linear'. ")
        kernel_tensor = self._construct_kernel(in_channels, kernel_base1d, nd)
        self.register_buffer("kernel", kernel_tensor, persistent=False)
        self.conv = [ff.conv_transpose1d, ff.conv_transpose2d, ff.conv_transpose3d][nd-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x, self.kernel, padding=self.padding, stride=self.scale, output_padding=self.outpadding)

    @staticmethod
    def _construct_kernel(in_channels: int, kernel_base1d: torch.Tensor, nd: int) -> torch.Tensor:
        kernel_size = len(kernel_base1d)
        # a disgusting substitute of "tensor.unsqueeze(-1).unsqueeze(-1)... n-1 times" syntax
        kernel_base1d = kernel_base1d[(...,) + (None,)*(nd-1)]
        kernel_base = kernel_base1d
        for i in range(1, nd):
            kernel_base = kernel_base * kernel_base1d.transpose(0, i)
        kernel = torch.zeros((in_channels,)*2 + (kernel_size,)*nd, dtype=torch.float32)
        for i in range(in_channels):
            kernel[i, i, ...] = kernel_base
        return kernel

    @staticmethod
    def _kernel_base_linear(scale: int) -> torch.Tensor:
        # The following kernel provides preservation of input points in the rescaled output but leaves one-sided border
        # artifacts.
        # palindrome = (torch.cat([torch.arange(1, scale+1), torch.arange(scale-1, 0, -1)]).float()) / scale

        # application of kernels below are meant to replace
        # nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False), as they provide identical internal
        # tensor elements.
        # The difference between ConvTranspose with these kernels and nn.Upsample lies in conv's zero padding. I guess,
        # if pytorch somehow supported 'reflect' padding for ConvTranspose, results at the borders would be identical
        # aswell.
        # Anyway, this implementation seems legit to me.
        if scale % 2 == 0:
            palindrome = torch.cat((torch.arange(1, scale*2, 2), torch.arange(2*scale-1, 0, -2))).float()
        else:
            palindrome = torch.cat((torch.arange(2, scale*2+1, 2), torch.arange(2*scale-2, 0, -2))).float()
        palindrome /= (2*scale)
        return palindrome

    @staticmethod
    def _kernel_base_nearest(scale: int):
        kbn = torch.zeros(2*scale-1)
        kbn[(scale-1):] = 1.
        return kbn


class Upsample1d(UpsampleNd):
    def __init__(self, in_channels: int, scale: int = 2, mode: str = "nearest"):
        super().__init__(in_channels, scale, mode, nd=1)


class Upsample2d(UpsampleNd):
    def __init__(self, in_channels: int, scale: int = 2, mode: str = "nearest"):
        super().__init__(in_channels, scale, mode, nd=2)


class Upsample3d(UpsampleNd):
    def __init__(self, in_channels: int, scale: int = 2, mode: str = "nearest"):
        super().__init__(in_channels, scale, mode, nd=3)


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
    def __init__(self, original_spatial_size: int, dim_from_batch: int = 2):
        super().__init__()
        self.orig_sp_size = original_spatial_size
        spatials_dim_idxs = [3, 4]
        spatials_dim_idxs.insert(dim_from_batch - 2, 1)
        self.perm_dims = [0, 2, *spatials_dim_idxs]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # restoring original dimensions
        # [B*X, C, Y, Z] -> [B, X, C, Y, Z] -> [B, C, X, Y, Z]
        # op = x.view(sz[0], sz[2], *x.shape[1:]).transpose(1, 2)
        op = x.view(-1, self.orig_sp_size, *x.shape[1:]).permute(self.perm_dims)   # maybe .contiguous() ?
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
    """ A simple encoder-decoder with 2d-3d-2d convolutions """
    def __init__(self, spatial_size: int, encoder2d: nn.Module, bottleneck3d: nn.Module, decoder2d: nn.Module,
                 leaping_dim: int = 2):
        super().__init__(
            To2d(dim_to_batch=leaping_dim),
            encoder2d,
            To3d(spatial_size, dim_from_batch=leaping_dim),
            bottleneck3d,
            To2d(dim_to_batch=leaping_dim),
            decoder2d,
            To3d(spatial_size, dim_from_batch=leaping_dim)
        )


def _t31(k: int, i: int, defval: int = 1) -> Tuple[int, ...]:
    t = [defval, defval]
    t.insert(i, k)
    return tuple(t)


def construct_bottleneck3d(n_chan: int, use_norm: bool = True, leaping_dim: int = 2):
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
    return bottleneck3d


def conv232_assembly(n_chan: int, spatial_size: int, use_norm: bool = True, leaping_dim: int = 2,
                     use_conadjust: bool = True):
    maybe_conadjust = ContrastAdjustment2d(4, 2) if use_conadjust else nn.Identity()
    encoder2d = nn.Sequential(
        maybe_conadjust,
        nn.Conv2d(4, n_chan, kernel_size=t2(3), padding=1, bias=True),
        nn.ReLU(inplace=True),
        ResLayer2d(n_chan, resampling=0, use_norm=use_norm),
        ResLayer2d(n_chan, resampling=-1, use_norm=use_norm),
        ResLayer2d(2*n_chan, resampling=-1, use_norm=use_norm),
    )
    bottleneck3d = construct_bottleneck3d(n_chan, use_norm, leaping_dim)
    decoder2d = nn.Sequential(
        ResLayer2d(4*n_chan, resampling=1, use_norm=use_norm),
        ResLayer2d(2*n_chan, resampling=1, use_norm=use_norm),
        ResLayer2d(n_chan, resampling=0, use_norm=use_norm),
        nn.Conv2d(n_chan, 4, kernel_size=t2(3), padding=1, bias=True)
    )
    model = Conv232d(spatial_size, encoder2d, bottleneck3d, decoder2d, leaping_dim=leaping_dim)
    return model


"""    #################################    """
"""    #          Conv232 Unet         #    """
"""    #################################    """


class Conv232Unet(nn.Module):
    def __init__(self, n_chan: int, spatial_size: int, kernel_size: int = 3, use_norm: bool = True,
                 leaping_dim: int = 2, skip_conn_op: str = 'cat'):
        super().__init__()
        if skip_conn_op == 'cat':
            self.skip_conn_func = torch.add
            skip_chan_mult = 1
        elif skip_conn_op == 'sum':
            self.skip_conn_func = lambda x, y: torch.cat((x, y), dim=1)
            skip_chan_mult = 2
        else:
            raise ValueError("Only 'cat' or 'sum' are available as skip_conn_op argument")
        self.to2d = To2d(dim_to_batch=leaping_dim)
        self.to3d = To3d(spatial_size, dim_from_batch=leaping_dim)
        self.proj_in = nn.Sequential(
            nn.Conv2d(4, n_chan, kernel_size=t2(kernel_size), padding=kernel_size//2, bias=True),
            nn.ReLU(inplace=True),
        )
        self.d0 = ResLayer2d(n_chan, kernel_size=kernel_size, resampling=0, use_norm=use_norm)
        self.d2 = ResLayer2d(n_chan, kernel_size=kernel_size, resampling=-1, use_norm=use_norm)
        self.d4 = ResLayer2d(2*n_chan, kernel_size=kernel_size, resampling=-1, use_norm=use_norm)
        self.bottleneck3d = construct_bottleneck3d(n_chan, use_norm, leaping_dim)
        self.u4 = ResLayer2d(skip_chan_mult*4*n_chan, 2*n_chan, kernel_size=kernel_size, resampling=1,
                             use_norm=use_norm)
        self.u2 = ResLayer2d(skip_chan_mult*2*n_chan, n_chan, kernel_size=kernel_size, resampling=1,
                             use_norm=use_norm)
        self.u0 = ResLayer2d(skip_chan_mult*n_chan, n_chan, kernel_size=kernel_size, resampling=0,
                             use_norm=use_norm)
        self.proj_out = nn.Conv2d(n_chan, 4, kernel_size=t2(kernel_size), padding=kernel_size//2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.d0(self.proj_in(self.to2d(x)))
        x2 = self.d2(x0)
        x4 = self.d4(x2)
        x4 = self.to3d(x4)
        x4 = self.skip_conn_func(x4, self.bottleneck3d(x4))
        x4 = self.to2d(x4)
        x2 = self.skip_conn_func(x2, self.u4(x4))
        x0 = self.skip_conn_func(x0, self.u2(x2))
        x0 = self.to3d(self.proj_out(self.u0(x0)))
        return x0


"""    ######################################    """
"""    #          Conv232 RefineNet         #    """
"""    ######################################    """


class RCU2d(nn.Module):
    def __init__(self, in_chan: int, mid_chan: Optional[int] = None, kernel_size: int = 3, n_blocks: int = 2):
        super().__init__()
        if mid_chan is None:
            mid_chan = in_chan
        self.n_blocks = n_blocks
        self.conv1list = nn.ModuleList(nn.Conv2d(in_chan, mid_chan, kernel_size=t2(kernel_size), padding=kernel_size//2)
                                       for _ in range(n_blocks))
        self.conv2list = nn.ModuleList(nn.Conv2d(mid_chan, in_chan, kernel_size=t2(kernel_size), padding=kernel_size//2)
                                       for _ in range(n_blocks))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.conv1list, self.conv2list):
            x += self.relu(conv2(self.relu(conv1(x))))
        return x


class MultiResolutionFusion(nn.Module):
    def __init__(self, in_chans: List[int], out_chan: int, scale_factors: List[int], upsample_mode="nearest"):
        super().__init__()
        self.upsamples = nn.ModuleList(nn.Identity() if sf == 1 else Upsample2d(out_chan, scale=sf, mode=upsample_mode)
                                       for sf in scale_factors)
        self.convs = nn.ModuleList(nn.Conv2d(inch, out_chan, kernel_size=t2(3), padding=1, bias=True)
                                   for inch in in_chans)

    def forward(self, multiscale_imgs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return sum(upsample(conv(img)) for upsample, conv, img in zip(self.upsamples, self.convs, multiscale_imgs))


class Conv232RefineNet(nn.Module):
    def __init__(self, n_chan: int, spatial_size: int, kernel_size: int = 3, use_norm: bool = True,
                 leaping_dim: int = 2):
        super().__init__()
        self.to2d = To2d(dim_to_batch=leaping_dim)
        self.to3d = To3d(spatial_size, dim_from_batch=leaping_dim)
        self.proj_in = nn.Sequential(
            nn.Conv2d(4, n_chan, kernel_size=t2(kernel_size), padding=kernel_size//2, bias=True),
            nn.ReLU(inplace=True),
        )
        self.d0 = ResLayer2d(n_chan, kernel_size=kernel_size, resampling=0, use_norm=use_norm)
        self.d2 = ResLayer2d(n_chan, kernel_size=kernel_size, resampling=-1, use_norm=use_norm)
        self.d4 = ResLayer2d(2*n_chan, kernel_size=kernel_size, resampling=-1, use_norm=use_norm)
        self.bottleneck3d = construct_bottleneck3d(n_chan, use_norm, leaping_dim)
        self.rcu0 = RCU2d(n_chan, n_blocks=2)
        self.rcu2 = RCU2d(2*n_chan, n_blocks=2)
        self.rcu4 = RCU2d(4*n_chan, n_blocks=2)
        self.merge = MultiResolutionFusion(in_chans=[n_chan, 2*n_chan, 4*n_chan], out_chan=n_chan,
                                           scale_factors=[1, 2, 4], upsample_mode="nearest")
        self.u0 = ResLayer2d(n_chan, kernel_size=kernel_size, resampling=0, use_norm=use_norm)
        self.proj_out = nn.Conv2d(n_chan, 4, kernel_size=t2(kernel_size), padding=kernel_size//2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.d0(self.proj_in(self.to2d(x)))
        x2 = self.d2(x0)
        x4 = self.d4(x2)
        x4 = self.to3d(x4)
        x4 = x4 + self.bottleneck3d(x4)
        x4 = self.to2d(x4)
        x0 = self.merge((x0, x2, x4))
        x0 = self.to3d(self.proj_out(self.u0(x0)))
        return x0


class Conv232RefineNetCascade(Conv232RefineNet):
    def __init__(self, n_chan: int, spatial_size: int, kernel_size: int = 3, use_norm: bool = True,
                 leaping_dim: int = 2):
        super().__init__(n_chan, spatial_size, kernel_size, use_norm, leaping_dim)
        self.u2 = ResLayer2d(2*n_chan, kernel_size=kernel_size, resampling=0, use_norm=use_norm)
        self.u0 = ResLayer2d(n_chan, kernel_size=kernel_size, resampling=0, use_norm=use_norm)
        self.merge = MultiResolutionFusion(in_chans=[n_chan, 2*n_chan], out_chan=n_chan,
                                           scale_factors=[1, 2], upsample_mode="nearest")
        self.merge2 = MultiResolutionFusion(in_chans=[2*n_chan, 4*n_chan], out_chan=2*n_chan,
                                            scale_factors=[1, 2], upsample_mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.d0(self.proj_in(self.to2d(x)))
        x2 = self.d2(x0)
        x4 = self.d4(x2)
        x4 = self.to3d(x4)
        x4 = x4 + self.bottleneck3d(x4)
        x4 = self.to2d(x4)
        x2 = self.merge2((x2, x4))
        x2 = self.u2(x2)
        x0 = self.merge((x0, x2))
        x0 = self.to3d(self.proj_out(self.u0(x0)))
        return x0


"""    #############################    """


if __name__ == "__main__":

    tmp_in = torch.randn((2, 4, 128, 128, 80))
    op2 = SImple(8)(tmp_in)
    print(op2.shape)

    tmp_model = conv232_assembly(n_chan=3, spatial_size=128)
    op2 = tmp_model(tmp_in)
    print(op2.shape)

    tmp_model = Conv232Unet(n_chan=3, spatial_size=128)
    op2 = tmp_model(tmp_in)
    print(op2.shape)

    tmp_model = Conv232RefineNet(n_chan=3, spatial_size=128)
    op2 = tmp_model(tmp_in)
    print(op2.shape)

    tmp_model = Conv232RefineNetCascade(n_chan=3, spatial_size=128)
    op2 = tmp_model(tmp_in)
    print(op2.shape)

    print("woah!")
