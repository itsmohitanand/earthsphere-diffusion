import torch
import torch.nn as nn
from torch.nn import Module
from math import sqrt
from einops import rearrange
import math

import torch.nn.functional as F

from es.backbone.utils import normalize_weight, divisible_by, l2norm, pack_one, unpack_one


class Conv2d(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        eps=1e-4,
        concat_ones_to_input=False,  # they use this in the input block to protect against loss of expressivity due to removal of all biases, even though they claim they observed none
    ):
        super().__init__()
        weight = torch.randn(
            dim_out, dim_in + int(concat_ones_to_input), kernel_size, kernel_size
        )
        self.weight = nn.Parameter(weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size**2
        self.concat_ones_to_input = concat_ones_to_input
        self.pad_width = (kernel_size - 1) // 2

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps=self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps=self.eps) / sqrt(self.fan_in)

        if self.concat_ones_to_input:
            x = F.pad(x, (0, 0, 0, 0, 1, 0), value=1.0)

        # periodic padding
        if self.pad_width > 0:
            # periodic padding on the longitude dimension (width)
            x = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular")
            # zero padding on the latitude dimension (height)
            x = F.pad(
                x, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0
            )

        return F.conv2d(x, weight, padding="valid")

class Linear(Module):
    def __init__(self, dim_in, dim_out, eps=1e-4):
        super().__init__()
        weight = torch.randn(dim_out, dim_in)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps=self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps=self.eps) / sqrt(self.fan_in)
        return F.linear(x, weight)

class MPFourierEmbedding(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=False)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        return torch.cat((freqs.sin(), freqs.cos()), dim=-1) * sqrt(2)

class MPSiLU(Module):
    def forward(self, x):
        return F.silu(x) / 0.596


# gain - layer scaling


class Gain(Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x * self.gain


# magnitude preserving concat
# equation (103) - default to 0.5, which they recommended


class MPCat(Module):
    def __init__(self, t=0.5, dim=-1):
        super().__init__()
        self.t = t
        self.dim = dim

    def forward(self, a, b):
        dim, t = self.dim, self.t
        Na, Nb = a.shape[dim], b.shape[dim]

        C = sqrt((Na + Nb) / ((1.0 - t) ** 2 + t**2))

        a = a * (1.0 - t) / sqrt(Na)
        b = b * t / sqrt(Nb)

        return C * torch.cat((a, b), dim=dim)


# magnitude preserving sum
# equation (88)
# empirically, they found t=0.3 for encoder / decoder / attention residuals
# and for embedding, t=0.5


class MPAdd(Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x, res):
        a, b, t = x, res, self.t
        num = a * (1.0 - t) + b * t
        den = sqrt((1 - t) ** 2 + t**2)
        return num / den


# pixelnorm
# equation (30)


class PixelNorm(Module):
    def __init__(self, dim, eps=1e-4):
        super().__init__()
        # high epsilon for the pixel norm in the paper
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        dim = self.dim
        return l2norm(x, dim=dim, eps=self.eps) * sqrt(x.shape[dim])
