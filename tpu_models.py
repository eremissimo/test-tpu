import torch
import torch.nn as nn
import torch.nn.functional as ff
from torchmetrics.functional.classification import recall
from padding import get_padding
from typing import Tuple, Optional, List


"""    #############################    """
"""    #          Losses           #    """
"""    #############################    """
"""TODO: move losses to a separate file """


def focal_loss(input: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None,
               gamma: float = 1.0) -> torch.Tensor:
    weight[weight.isnan()] = 0.
    weight /= weight.sum()
    ce = ff.cross_entropy(input, target, reduction="none")
    probs = torch.exp(-ce)
    loss = ((1 - probs).pow(gamma))*ce
    if weight is not None:
        loss *= weight[target]
    loss_reduced = loss.mean(dim=0).sum()
    return loss_reduced


def recall_ce_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """From the unpublished article "Recall Loss for Imbalanced Image Classification and Semantic Segmentation" """
    batch_recall = recall(input, target, average='none', num_classes=input.shape[1], mdmc_average='global')
    batch_recall[batch_recall.isnan()] = 1.
    return ff.cross_entropy(input, target, weight=(1. - batch_recall))


def soft_iou_loss(input: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    probs = input.softmax(dim=1)
    targ_probs = ff.one_hot(target, num_classes=input.shape[1]).permute([0, 4, 1, 2, 3])
    intersection = (probs * targ_probs).sum(dim=[2, 3, 4])
    union = (probs + targ_probs - probs * targ_probs).sum(dim=[2, 3, 4])
    iou = intersection/union
    if weight is not None:
        weight /= weight.sum()
        # unsqueezing for broadcasting
        iou *= weight[None, :, None, None, None]
    return 1.0 - iou.mean()


def hausdorff_loss(input: torch.Tensor, target: torch.Tensor, dt: torch.Tensor, weight: Optional[torch.Tensor] = None) \
        -> torch.Tensor:
    """One-sided hausdorff loss from the article:
     Davood Karimi, Septimiu E. Salcudean "Reducing the Hausdorff Distance in Medical Image Segmentation with
     Convolutional Neural Networks"
     https://arxiv.org/abs/1904.10030
    """
    target = ff.one_hot(target, num_classes=input.shape[1]).permute([0, 4, 1, 2, 3])
    input = input.softmax(dim=1)
    loss = (input - target).square() * dt
    if weight is not None:
        weight /= weight.sum()
        # unsqueezing for broadcasting
        loss *= weight[None, :, None, None, None]
    return loss.mean()


class HausdorffErosion3d(nn.Module):
    """ From the article:
     Davood Karimi, Septimiu E. Salcudean "Reducing the Hausdorff Distance in Medical Image Segmentation with
     Convolutional Neural Networks"
     https://arxiv.org/abs/1904.10030
     """
    def __init__(self, n_classes: int, n_steps: int = 10, threshold: float = 0.7, alpha: float = 2.0,
                 to_probs: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.n_steps = n_steps
        self.alpha = alpha
        cross = torch.zeros((3, 3, 3), dtype=torch.float32, requires_grad=False)
        cross[1, 1, :] = 1.
        cross[1, :, 1] = 1.
        cross[:, 1, 1] = 1.
        self.kernel = (1./7./(1.-threshold))*cross.repeat((n_classes, 1, 1, 1, 1))
        self.bias = torch.empty((n_classes,), dtype=torch.float32, requires_grad=False)
        self.bias.fill_(-threshold/(1. - threshold))
        self.convert_inputs_to_probs = to_probs

    def forward(self, input: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.convert_inputs_to_probs:
            input = input.softmax(dim=1)
        target = ff.one_hot(target, num_classes=self.n_classes).permute([0, 4, 1, 2, 3])
        op = (input - target).square()
        # erosion as convolution + thresholding
        # soft tresholding can be viewed as a composition of a linear transform and relu
        # linear part of that is fused with conv (kernel coefficient and bias)
        loss_t = torch.zeros_like(input)
        for k in range(self.n_steps):
            op = ff.conv3d(op, self.kernel, bias=self.bias, padding=(1, 1, 1), groups=self.n_classes)
            op.relu_()
            loss_t += (float(k+1)**self.alpha)*op
        if weight is not None:
            # this dirty syntax represents even more stupid
            # weight.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            # TODO: check if it is supported by xla without a context switch
            weight_t = weight[(...,)+(None,)*3]
            loss_t *= weight_t
        # loss = (1./self.n_classes)*loss_t.sum()
        loss = loss_t.mean()
        return loss


"""    #############################    """
"""    #          Layers           #    """
"""    #############################    """
"""TODO: move layers to a separate file """


def t3(val):
    return val, val, val


def t2(val):
    return val, val


def t1(val):
    return val,


class ResLayer(nn.Module):
    """ ResNet Layer """
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, kernel_size: int = 3, resampling: int = 0,
                 use_norm: bool = True, groups: int = 1, ndim: int = 3):
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
                                  stride=tn(stride), bias=False, groups=groups)
        else:
            resample_layer = Convtr(in_channels, mid_channels, kernel_size=tn(ct_kernel_size),
                                    padding=ct_padding, stride=tn(2), bias=False, groups=groups)
        module_list = []
        if use_norm:
            module_list.append(Bn(in_channels))
        module_list.append(Conv(in_channels, in_channels, kernel_size=tn(kernel_size),
                                padding=(kernel_size//2), bias=False, groups=groups))
        module_list.append(nn.SELU(inplace=True))
        if use_norm:
            module_list.append(Bn(in_channels))
        module_list.append(resample_layer)
        self.main_branch = nn.Sequential(*module_list)

        if resampling < 0:
            self.skip_branch = Conv(in_channels, mid_channels, kernel_size=tn(1), stride=tn(2), bias=False,
                                    groups=groups)
        elif resampling == 0:
            self.skip_branch = nn.Identity()
        else:
            self.skip_branch = Convtr(in_channels, mid_channels, kernel_size=tn(2), stride=tn(2),
                                      bias=False, groups=groups)
        self.selu = nn.SELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        op = self.main_branch(x) + self.skip_branch(x)
        op = self.selu(op)
        return op


class ResLayer1d(ResLayer):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, kernel_size: int = 3, resampling: int = 0,
                 use_norm: bool = True, groups: int = 1):
        super().__init__(in_channels, out_channels, kernel_size, resampling, use_norm, groups, ndim=1)


class ResLayer2d(ResLayer):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, kernel_size: int = 3, resampling: int = 0,
                 use_norm: bool = True, groups: int = 1):
        super().__init__(in_channels, out_channels, kernel_size, resampling, use_norm, groups, ndim=2)


class ResLayer3d(ResLayer):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, kernel_size: int = 3, resampling: int = 0,
                 use_norm: bool = True, groups: int = 1):
        super().__init__(in_channels, out_channels, kernel_size, resampling, use_norm, groups, ndim=3)


class MultiResLayer(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, kernel_size: int = 3, resampling: int = 0,
                 use_norm: bool = True, groups: int = 1, n_layers: int = 2, ndim: int = 3):
        reslayers = []
        for i in range(n_layers):
            outch = out_channels if i == (n_layers - 1) else in_channels
            res = 0 if i < (n_layers - 1) else resampling
            reslayers.append(
                ResLayer(in_channels, outch, kernel_size=kernel_size, resampling=res, use_norm=use_norm, groups=groups,
                         ndim=ndim)
            )
        super().__init__(*reslayers)


class MultiResLayer3d(MultiResLayer):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None,
                 kernel_size: int = 3, resampling: int = 0, use_norm: bool = True, groups: int = 1, n_layers: int = 2):
        super().__init__(in_channels, out_channels, kernel_size, resampling, use_norm, groups, n_layers, ndim=3)


class MultiResLayer2d(MultiResLayer):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None,
                 kernel_size: int = 3, resampling: int = 0, use_norm: bool = True, groups: int = 1, n_layers: int = 2):
        super().__init__(in_channels, out_channels, kernel_size, resampling, use_norm, groups, n_layers, ndim=2)


class MultiResLayer1d(MultiResLayer):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None,
                 kernel_size: int = 3, resampling: int = 0, use_norm: bool = True, groups: int = 1, n_layers: int = 2):
        super().__init__(in_channels, out_channels, kernel_size, resampling, use_norm, groups, n_layers, ndim=1)


class UpsampleNd(nn.Module):
    """ Upsample as ConvTranspose with certain kernels. Interpolation is not supported by xla yet, that's why the
        native nn.Upsample is slow on TPU.
        BTW, only natural scale factors are supported."""
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

        # Application of kernels below is meant to replace
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
        ker01 = torch.zeros(2*scale-1)
        ker01[(scale-1):] = 1.
        return ker01


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
        op = x.permute(self.perm_dims).flatten(start_dim=0, end_dim=1)
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
        op = x.view(-1, self.orig_sp_size, *x.shape[1:]).permute(self.perm_dims)
        return op


class ContrastAdjustment(nn.Module):
    """ A simple elementwise remapping of the input featuremap x = f(x), where f is parametrized as a sum of the
        identity and gaussian terms with learnable parameters
        f(x) = x*(1 + sum(a[i] * N(x | m[i], s[i])))

        On init the module initializes as near-identity
    """
    def __init__(self, n_chan: int,  num_terms: int = 2, nd: int = 3, stds_lower_bound=0.1):
        super().__init__()
        # [terms, B, chan, X, Y, Z] broadcasting scheme (for 3d)
        shp = [num_terms, 1, n_chan] + [1]*nd
        self.std_bound = stds_lower_bound
        self.amplitudes = nn.Parameter(0.032*torch.randn(shp), requires_grad=True)
        self.means = nn.Parameter(2*torch.rand(shp), requires_grad=True)
        self.std_logits = nn.Parameter(torch.zeros(shp, dtype=torch.float32), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        invstds = (1./self.std_bound) * (self.std_logits.sigmoid())
        op = x*(1 + (self.amplitudes*torch.exp(-0.5*((x - self.means)*invstds).square())).sum(dim=0))
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


class ElementwiseTransform(nn.Module):
    """Nonlinear transformation of input channels"""
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net      # MLP

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, D3, D1, D2, C]
        x = x.transpose(1, 4)
        # [B, D3, D1, D2, C1]
        x = self.net(x)
        # [B, C1, D1, D2, D3]
        x = x.transpose(1, 4)
        return x


class ResidualMLP(nn.Module):
    """ MLP blocks with (linear) skip-connections """
    def __init__(self, n_in: int, n_mid: int, n_out: int, n_blocks: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.Linear(n_mid, n_mid),
            nn.SELU(),
            nn.Linear(n_mid, n_mid)
        ) for _ in range(n_blocks))
        self.blocks[0][0] = nn.Linear(n_in, n_mid)
        self.blocks[-1][-1] = nn.Linear(n_mid, n_out)
        self.projs = nn.ModuleList(nn.Identity() for _ in range(n_blocks))
        if n_blocks > 1:
            self.projs[0] = nn.Linear(n_in, n_mid)
            self.projs[-1] = nn.Linear(n_mid, n_out)
        elif n_in != n_out:
            self.projs[0] = nn.Linear(n_in, n_out)

    def forward(self, x: torch.Tensor):
        for proj, block in zip(self.projs, self.blocks):
            x = proj(x) + block(x)
        return x


class NewRefineHead(nn.Module):
    """Patch based probability map refinement.
                   ________________________________________[transform_net] ___ transformed_img_patch
    image_patch __/__ [context_net] ___ context_vectors__/                       \
    prob_patch ____/                                                              \
            \______________________________________________________________________\__[prob_refine] ___ refined_prob
    """
    def __init__(self, n_ctx: int = 16, n_trans: int = 8, n_steps: int = 1):
        super().__init__()
        self.n_ctx = n_ctx
        self.n_steps = n_steps
        self.ctx_net = ElementwiseTransform(ResidualMLP(8, 8, n_ctx, n_blocks=2))
        self.trans_net = ElementwiseTransform(ResidualMLP(4 + n_ctx, 8, n_trans, n_blocks=2))
        mean_ker = torch.ones((n_ctx, 1, 3, 3, 3), dtype=torch.float32) / 27.
        self.register_buffer("mean_ker", mean_ker, persistent=False)
        self.sharpness = nn.Parameter(torch.tensor(1.))
        self.shiftdims = [(shift, dim) for dim in (2, 3, 4) for shift in (-1, 1)]

    def forward(self, img: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_steps):
            ctx_map = self.ctx_net(torch.cat((img, probs), dim=1))
            ctx_map = ff.conv3d(ctx_map, self.mean_ker, padding=t3(1), groups=self.n_ctx)
            # [B, n_trans, D1, D2, D3]
            ft_map = self.trans_net(torch.cat((img, ctx_map), dim=1))
            # instead of computing a combination of probs based on featuremap similarity coefficients over 3*3*3
            # unfolded patches like in previous version of this class, here we compute probs and similarities only over
            # 6-neighbor voxels and without explicit unfolding. This should be a lot faster, not so accurate though.
            for ds in self.shiftdims:
                probs = probs + self._prob_similarity_shifted(probs, ft_map, *ds)
            probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    def _prob_similarity_shifted(self, probs: torch.Tensor, ft_map: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
        # compute p_j*exp( - sharpness * norm(ft_i - ft_j)^2) for a single shift
        probs_shifted = self._shift_tensor(probs, shift, dim)
        ft_shifted = self._shift_tensor(ft_map, shift, dim)
        similarity = torch.exp(- self.sharpness * (ft_shifted - ft_map).square().sum(dim=1, keepdim=True))
        probs_shifted = probs_shifted*similarity
        return probs_shifted

    @staticmethod
    def _shift_tensor(t: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
        n = t.shape[dim]
        sl = [slice(None)]*5        # t_in.dim() == 5
        sr = sl.copy()
        sh_eq_1 = int(shift == 1)
        sl[dim] = slice(0 + sh_eq_1, n - 1 + sh_eq_1)
        sr[dim] = slice(0 + 1 - sh_eq_1, n - sh_eq_1)
        sl = tuple(sl)
        sr = tuple(sr)
        t_shifted = torch.zeros_like(t)
        t_shifted[sl] = t[sr]
        return t_shifted


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
            nn.SELU(inplace=True),
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


def construct_bottleneck3d(n_chan: int, use_norm: bool = True, leaping_dim: int = 2, groups: int = 1):
    maybe_bn3d_1 = nn.BatchNorm3d(4*n_chan) if use_norm else nn.Identity()
    maybe_bn3d_2 = nn.BatchNorm3d(4*n_chan) if use_norm else nn.Identity()
    x_downsample = 4
    bottleneck3d = nn.Sequential(
        maybe_bn3d_1,
        nn.Conv3d(4*n_chan, 4*n_chan, kernel_size=_t31(x_downsample+1, leaping_dim-2),
                  stride=_t31(x_downsample, leaping_dim-2),
                  padding=_t31(x_downsample//2, leaping_dim-2, 0),
                  bias=False, groups=groups),
        nn.SELU(inplace=True),
        ResLayer3d(4*n_chan, resampling=0, use_norm=use_norm, groups=groups),
        ResLayer3d(4*n_chan, resampling=0, use_norm=use_norm, groups=groups),
        ResLayer3d(4*n_chan, resampling=0, use_norm=use_norm, groups=groups),
        maybe_bn3d_2,
        nn.ConvTranspose3d(4*n_chan, 4*n_chan, kernel_size=_t31(x_downsample+1, leaping_dim-2),
                           stride=_t31(x_downsample, leaping_dim-2),
                           padding=_t31(x_downsample//2 - 1, leaping_dim-2, 0),
                           output_padding=_t31(1, leaping_dim-2, 0), bias=False, groups=groups),
        nn.SELU(inplace=True)
    )
    return bottleneck3d


def conv232_assembly(n_chan: int, spatial_size: int, use_norm: bool = True, leaping_dim: int = 2,
                     use_conadjust: bool = True):
    maybe_conadjust = ContrastAdjustment2d(4, 2) if use_conadjust else nn.Identity()
    encoder2d = nn.Sequential(
        maybe_conadjust,
        nn.Conv2d(4, n_chan, kernel_size=t2(3), padding=1, bias=True),
        nn.SELU(inplace=True),
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
        ngroups = 4
        n_ca = 4
        self.to2d = To2d(dim_to_batch=leaping_dim)
        self.to3d = To3d(spatial_size, dim_from_batch=leaping_dim)
        self.contr_adjust = nn.ModuleList(ContrastAdjustment2d(4, num_terms=4) for _ in range(n_ca))
        self.proj_in = nn.Sequential(
            nn.Conv2d(4*n_ca, n_chan, kernel_size=t2(kernel_size), padding=kernel_size//2, bias=True, groups=ngroups),
            nn.SELU(inplace=True),
        )
        self.d0 = MultiResLayer2d(n_chan, kernel_size=kernel_size, resampling=0, use_norm=use_norm, groups=ngroups)
        self.d2 = MultiResLayer2d(n_chan, kernel_size=kernel_size, resampling=-1, use_norm=use_norm, groups=ngroups)
        self.d4 = MultiResLayer2d(2*n_chan, kernel_size=kernel_size, resampling=-1, use_norm=use_norm, groups=ngroups)
        self.bottleneck3d = construct_bottleneck3d(n_chan, use_norm, leaping_dim, groups=ngroups)
        self.u4 = MultiResLayer2d(skip_chan_mult*4*n_chan, 2*n_chan, kernel_size=kernel_size, resampling=1,
                                  use_norm=use_norm, groups=ngroups)
        self.u2 = MultiResLayer2d(skip_chan_mult*2*n_chan, n_chan, kernel_size=kernel_size, resampling=1,
                                  use_norm=use_norm, groups=ngroups)
        self.u0 = MultiResLayer2d(skip_chan_mult*n_chan, n_chan, kernel_size=kernel_size, resampling=0,
                                  use_norm=use_norm, groups=ngroups)
        self.proj_out = nn.Conv2d(n_chan, 4, kernel_size=t2(kernel_size), padding=kernel_size//2, bias=True,
                                  groups=ngroups)
        self.refine_head = NewRefineHead(n_trans=4, n_steps=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.to2d(x)
        x0 = torch.cat([contr_adj(x0) for contr_adj in self.contr_adjust], dim=1)
        x0 = self.d0(self.proj_in(x0))
        x2 = self.d2(x0)
        x4 = self.d4(x2)
        x4 = self.to3d(x4)
        x4 = self.skip_conn_func(x4, self.bottleneck3d(x4))
        x4 = self.to2d(x4)
        x2 = self.skip_conn_func(x2, self.u4(x4))
        x0 = self.skip_conn_func(x0, self.u2(x2))
        x0 = self.to3d(self.proj_out(self.u0(x0)))
        x0 = x0.softmax(dim=1)
        x0 = self.refine_head(x, x0)
        return x0


"""    ######################################    """
"""    #          Conv232 RefineNet         #    """
"""    ######################################    """


class RCU2d(nn.Module):
    def __init__(self, in_chan: int, mid_chan: Optional[int] = None, kernel_size: int = 3, groups: int = 1,
                 n_blocks: int = 2):
        super().__init__()
        if mid_chan is None:
            mid_chan = in_chan
        self.n_blocks = n_blocks
        self.conv1list = nn.ModuleList(nn.Conv2d(in_chan, mid_chan, kernel_size=t2(kernel_size), padding=kernel_size//2,
                                                 groups=groups)
                                       for _ in range(n_blocks))
        self.conv2list = nn.ModuleList(nn.Conv2d(mid_chan, in_chan, kernel_size=t2(kernel_size), padding=kernel_size//2,
                                                 groups=groups)
                                       for _ in range(n_blocks))
        self.selu = nn.SELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.conv1list, self.conv2list):
            x += self.selu(conv2(self.selu(conv1(x))))
        return x


class MultiResolutionFusion(nn.Module):
    def __init__(self, in_chans: List[int], out_chan: int, scale_factors: List[int], groups: int = 1,
                 upsample_mode="nearest"):
        super().__init__()
        self.upsamples = nn.ModuleList(nn.Identity() if sf == 1 else Upsample2d(out_chan, scale=sf, mode=upsample_mode)
                                       for sf in scale_factors)
        self.convs = nn.ModuleList(nn.Conv2d(inch, out_chan, kernel_size=t2(3), padding=1, bias=True, groups=groups)
                                   for inch in in_chans)

    def forward(self, multiscale_imgs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        return sum(upsample(conv(img)) for upsample, conv, img in zip(self.upsamples, self.convs, multiscale_imgs))


class Conv232RefineNet(nn.Module):
    def __init__(self, n_chan: int, spatial_size: int, kernel_size: int = 3, use_norm: bool = True,
                 leaping_dim: int = 2):
        super().__init__()
        ngroups = 4
        n_ca = 4
        self.to2d = To2d(dim_to_batch=leaping_dim)
        self.to3d = To3d(spatial_size, dim_from_batch=leaping_dim)
        self.contr_adjust = nn.ModuleList(ContrastAdjustment2d(4, num_terms=4) for _ in range(n_ca))
        self.proj_in = nn.Sequential(
            nn.Conv2d(4*n_ca, n_chan, kernel_size=t2(kernel_size), padding=kernel_size//2, bias=True, groups=ngroups),
            nn.SELU(inplace=True),
        )
        self.d0 = MultiResLayer2d(n_chan, kernel_size=kernel_size, resampling=0, use_norm=use_norm, groups=ngroups)
        self.d2 = MultiResLayer2d(n_chan, kernel_size=kernel_size, resampling=-1, use_norm=use_norm, groups=ngroups)
        self.d4 = MultiResLayer2d(2*n_chan, kernel_size=kernel_size, resampling=-1, use_norm=use_norm, groups=ngroups)
        self.bottleneck3d = construct_bottleneck3d(n_chan, use_norm, leaping_dim, groups=ngroups)
        self.rcu0 = RCU2d(n_chan, n_blocks=2, groups=ngroups)
        self.rcu2 = RCU2d(2*n_chan, n_blocks=2, groups=ngroups)
        self.rcu4 = RCU2d(4*n_chan, n_blocks=2, groups=ngroups)
        self.merge = MultiResolutionFusion(in_chans=[n_chan, 2*n_chan, 4*n_chan], out_chan=n_chan,
                                           scale_factors=[1, 2, 4], upsample_mode="nearest", groups=ngroups)
        self.u0 = MultiResLayer2d(n_chan, kernel_size=kernel_size, resampling=0, use_norm=use_norm, groups=ngroups)
        self.proj_out = nn.Conv2d(n_chan, 4, kernel_size=t2(kernel_size), padding=kernel_size//2, bias=True,
                                  groups=ngroups)
        self.refine_head = NewRefineHead(n_trans=4, n_steps=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.to2d(x)
        x0 = torch.cat([contr_adj(x0) for contr_adj in self.contr_adjust], dim=1)
        x0 = self.d0(self.proj_in(x0))
        x2 = self.d2(x0)
        x4 = self.d4(x2)
        x4 = self.to3d(x4)
        x4 = x4 + self.bottleneck3d(x4)
        x4 = self.to2d(x4)
        x0 = self.merge((x0, x2, x4))
        x0 = self.to3d(self.proj_out(self.u0(x0)))
        x0 = x0.softmax(dim=1)
        x0 = self.refine_head(x, x0)
        return x0


class Conv232RefineNetCascade(Conv232RefineNet):
    def __init__(self, n_chan: int, spatial_size: int, kernel_size: int = 3, use_norm: bool = True,
                 leaping_dim: int = 2):
        super().__init__(n_chan, spatial_size, kernel_size, use_norm, leaping_dim)
        ngroups = 4
        self.u2 = MultiResLayer2d(2*n_chan, kernel_size=kernel_size, resampling=0, use_norm=use_norm, groups=ngroups)
        self.u0 = MultiResLayer2d(n_chan, kernel_size=kernel_size, resampling=0, use_norm=use_norm, groups=ngroups)
        self.merge = MultiResolutionFusion(in_chans=[n_chan, 2*n_chan], out_chan=n_chan,
                                           scale_factors=[1, 2], upsample_mode="nearest", groups=ngroups)
        self.merge2 = MultiResolutionFusion(in_chans=[2*n_chan, 4*n_chan], out_chan=2*n_chan,
                                            scale_factors=[1, 2], upsample_mode="nearest", groups=ngroups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.to2d(x)
        x0 = torch.cat([contr_adj(x0) for contr_adj in self.contr_adjust], dim=1)
        x0 = self.d0(self.proj_in(x0))
        x2 = self.d2(x0)
        x4 = self.d4(x2)
        x4 = self.to3d(x4)
        x4 = x4 + self.bottleneck3d(x4)
        x4 = self.to2d(x4)
        x2 = self.merge2((x2, x4))
        x2 = self.u2(x2)
        x0 = self.merge((x0, x2))
        x0 = self.to3d(self.proj_out(self.u0(x0)))
        x0 = x0.softmax(dim=1)
        x0 = self.refine_head(x, x0)
        return x0


"""    #############################    """


if __name__ == "__main__":

    targ = torch.randint(4, (2, 64, 64, 40))
    tmp_in = torch.randn((2, 4, 64, 64, 40))
    op2 = SImple(8)(tmp_in)
    print(op2.shape)

    tmp_model = conv232_assembly(n_chan=4, spatial_size=64)
    op2 = tmp_model(tmp_in)
    print(op2.shape)

    tmp_model = Conv232Unet(n_chan=4, spatial_size=64)
    op2 = tmp_model(tmp_in)
    print(op2.shape)

    tmp_model = Conv232RefineNet(n_chan=4, spatial_size=64)
    op2 = tmp_model(tmp_in)
    print(op2.shape)

    tmp_model = Conv232RefineNetCascade(n_chan=4, spatial_size=64)
    op2 = tmp_model(tmp_in)
    print(op2.shape)

    tmp_model = NewRefineHead(n_steps=2)
    tmp_probs = tmp_in.softmax(dim=1)
    op2 = tmp_model(tmp_in, tmp_probs)
    print(op2.shape)

    loss = HausdorffErosion3d(4, n_steps=10)(op2, targ)
    print(loss)


    print("woah!")
