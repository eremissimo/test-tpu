""" Padding Helpers
from this repo:
https://github.com/rwightman/pytorch-image-models
"""


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding
