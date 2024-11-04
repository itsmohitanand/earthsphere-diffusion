import torch
import torch.nn as nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from functools import partial
from math import ceil, sqrt

import torch.nn.functional as F

from es.backbone.attend import Attend

from es.backbone.base_layers import Conv2d, Linear, PixelNorm, MPAdd, MPSiLU, Gain, MPFourierEmbedding, MPCat
from es.backbone.attend import Attend
from es.backbone.utils import exists, default, cast_tuple, prepend, append, xnor

class Encoder(Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        *,
        emb_dim=None,
        dropout=0.1,
        mp_add_t=0.3,
        has_attn=False,
        attn_dim_head=64,
        attn_res_mp_add_t=0.3,
        attn_flash=False,
        downsample=False,
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.downsample = downsample
        self.downsample_conv = None

        curr_dim = dim
        if downsample:
            self.downsample_conv = Conv2d(curr_dim, dim_out, 1)
            curr_dim = dim_out

        self.pixel_norm = PixelNorm(dim=1)

        self.to_emb = None
        if exists(emb_dim):
            self.to_emb = nn.Sequential(Linear(emb_dim, dim_out), Gain())

        self.block1 = nn.Sequential(MPSiLU(), Conv2d(curr_dim, dim_out, 3))

        self.block2 = nn.Sequential(
            MPSiLU(), nn.Dropout(dropout), Conv2d(dim_out, dim_out, 3)
        )

        self.res_mp_add = MPAdd(t=mp_add_t)

        self.attn = None
        if has_attn:
            self.attn = Attention(
                dim=dim_out,
                heads=max(ceil(dim_out / attn_dim_head), 2),
                dim_head=attn_dim_head,
                mp_add_t=attn_res_mp_add_t,
                flash=attn_flash,
            )

    def forward(self, x, emb=None):
        if self.downsample:
            h, w = x.shape[-2:]
            x = F.interpolate(x, (h // 2, w // 2), mode="bilinear")
            x = self.downsample_conv(x)

        x = self.pixel_norm(x)

        res = x.clone()

        x = self.block1(x)

        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, "b c -> b c 1 1")

        x = self.block2(x)

        x = self.res_mp_add(x, res)

        if exists(self.attn):
            x = self.attn(x)

        return x


class Decoder(Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        *,
        emb_dim=None,
        dropout=0.1,
        mp_add_t=0.3,
        has_attn=False,
        attn_dim_head=64,
        attn_res_mp_add_t=0.3,
        attn_flash=False,
        upsample=False,
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.upsample = upsample
        self.needs_skip = not upsample

        self.to_emb = None
        if exists(emb_dim):
            self.to_emb = nn.Sequential(Linear(emb_dim, dim_out), Gain())

        self.block1 = nn.Sequential(MPSiLU(), Conv2d(dim, dim_out, 3))

        self.block2 = nn.Sequential(
            MPSiLU(), nn.Dropout(dropout), Conv2d(dim_out, dim_out, 3)
        )

        self.res_conv = Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.res_mp_add = MPAdd(t=mp_add_t)

        self.attn = None
        if has_attn:
            self.attn = Attention(
                dim=dim_out,
                heads=max(ceil(dim_out / attn_dim_head), 2),
                dim_head=attn_dim_head,
                mp_add_t=attn_res_mp_add_t,
                flash=attn_flash,
            )

    def forward(self, x, emb=None):
        if self.upsample:
            h, w = x.shape[-2:]
            x = F.interpolate(x, (h * 2, w * 2), mode="bilinear")

        res = self.res_conv(x)

        x = self.block1(x)

        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, "b c -> b c 1 1")

        x = self.block2(x)

        x = self.res_mp_add(x, res)

        if exists(self.attn):
            x = self.attn(x)

        return x


# attention


class Attention(Module):
    def __init__(
        self, dim, heads=4, dim_head=64, num_mem_kv=4, flash=False, mp_add_t=0.3
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.pixel_norm = PixelNorm(dim=-1)

        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = Conv2d(dim, hidden_dim * 3, 1)
        self.to_out = Conv2d(hidden_dim, dim, 1)

        self.mp_add = MPAdd(t=mp_add_t)

    def forward(self, x):
        res, b, c, h, w = x, *x.shape

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv
        )

        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        q, k, v = map(self.pixel_norm, (q, k, v))

        out = self.attend(q, k, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        out = self.to_out(out)

        return self.mp_add(out, res)


# unet proposed by karras
# bias-less, no group-norms, with magnitude preserving operations


class KarrasUnet(Module):
    """
    going by figure 21. config G
    """

    def __init__(
        self,
        *,
        dim=192,
        dim_max=768,  # channels will double every downsample and cap out to this value
        num_classes=None,  # in paper, they do 1000 classes for a popular benchmark
        channels=30,
        conditioning_channels=30,
        input_size=(112, 240),
        num_downsamples=3,
        num_blocks_per_stage=4,
        attn_res=(28, 14),  # height resolution that attention applies to
        fourier_dim=16,
        attn_dim_head=64,
        attn_flash=False,
        mp_cat_t=0.5,
        mp_add_emb_t=0.5,
        attn_res_mp_add_t=0.3,
        resnet_mp_add_t=0.3,
        dropout=0.1,
        self_condition=False,
    ):
        super().__init__()

        self.self_condition = self_condition

        # determine dimensions

        self.channels = channels
        self.conditioning_channels = conditioning_channels
        input_channels = channels * (2 if self_condition else 1) + conditioning_channels
        # we add an extra channel of ones to the input block to act as a bias

        # input and output blocks

        self.input_block = Conv2d(input_channels, dim, 3, concat_ones_to_input=True)

        self.output_block = nn.Sequential(Conv2d(dim, channels, 3), Gain())

        # time embedding

        emb_dim = dim * 4

        self.to_time_emb = nn.Sequential(
            MPFourierEmbedding(fourier_dim), Linear(fourier_dim, emb_dim)
        )

        # class embedding

        self.needs_class_labels = exists(num_classes)
        self.num_classes = num_classes

        if self.needs_class_labels:
            self.to_class_emb = Linear(num_classes, 4 * dim)
            self.add_class_emb = MPAdd(t=mp_add_emb_t)

        # final embedding activations

        self.emb_activation = MPSiLU()

        # number of downsamples

        self.num_downsamples = num_downsamples

        # attention

        attn_res = set(cast_tuple(attn_res))

        # resnet block

        block_kwargs = dict(
            dropout=dropout,
            emb_dim=emb_dim,
            attn_dim_head=attn_dim_head,
            attn_res_mp_add_t=attn_res_mp_add_t,
            attn_flash=attn_flash,
        )

        # unet encoder and decoders

        self.downs = ModuleList([])
        self.ups = ModuleList([])

        curr_dim = dim

        self.skip_mp_cat = MPCat(t=mp_cat_t, dim=1)

        # take care of skip connection for initial input block and first three encoder blocks

        prepend(self.ups, Decoder(dim * 2, dim, **block_kwargs))

        assert num_blocks_per_stage >= 1

        for _ in range(num_blocks_per_stage):
            enc = Encoder(curr_dim, curr_dim, **block_kwargs)
            dec = Decoder(curr_dim * 2, curr_dim, **block_kwargs)

            append(self.downs, enc)
            prepend(self.ups, dec)

        # stages

        for _ in range(self.num_downsamples):
            dim_out = min(dim_max, curr_dim * 2)
            has_attn = input_size[0] in attn_res
            upsample = Decoder(
                dim_out, curr_dim, has_attn=has_attn, upsample=True, **block_kwargs
            )

            input_size = (input_size[0] // 2, input_size[1] // 2)
            has_attn = input_size[0] in attn_res

            downsample = Encoder(
                curr_dim, dim_out, downsample=True, has_attn=has_attn, **block_kwargs
            )

            append(self.downs, downsample)
            prepend(self.ups, upsample)
            prepend(
                self.ups,
                Decoder(dim_out * 2, dim_out, has_attn=has_attn, **block_kwargs),
            )

            for _ in range(num_blocks_per_stage):
                enc = Encoder(dim_out, dim_out, has_attn=has_attn, **block_kwargs)
                dec = Decoder(dim_out * 2, dim_out, has_attn=has_attn, **block_kwargs)

                append(self.downs, enc)
                prepend(self.ups, dec)

            curr_dim = dim_out

        # take care of the two middle decoders

        mid_has_attn = input_size[0] in attn_res

        self.mids = ModuleList(
            [
                Decoder(curr_dim, curr_dim, has_attn=mid_has_attn, **block_kwargs),
                Decoder(curr_dim, curr_dim, has_attn=mid_has_attn, **block_kwargs),
            ]
        )

        self.out_dim = channels

    @property
    def downsample_factor(self):
        return 2**self.num_downsamples

    def forward(self, x, time, self_cond=None, class_labels=None):
        # validate image shape

        assert (
            x.shape[1] == self.channels + self.conditioning_channels
        ), f"{x.shape[1]}!= {self.channels} + {self.conditioning_channels}"  # self conditioning

        if self.self_condition:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((self_cond, x), dim=1)
        else:
            assert not exists(self_cond)

        # # add a channel of ones - I can remove this as it is actually added later in the input block
        # batch_size, _, h, w = x.shape
        # ones_channel = torch.ones(batch_size, 1, h, w, device=x.device, dtype=x.dtype)
        # x = torch.cat((x, ones_channel), dim=1)

        # time condition
        if len(time.shape) == 4:
            time = time[:, 0, 0, 0]
        time_emb = self.to_time_emb(time)

        # class condition

        assert xnor(exists(class_labels), self.needs_class_labels)

        if self.needs_class_labels:
            if class_labels.dtype in (torch.int, torch.long):
                class_labels = F.one_hot(class_labels, self.num_classes)

            assert class_labels.shape[-1] == self.num_classes
            class_labels = class_labels.float() * sqrt(self.num_classes)

            class_emb = self.to_class_emb(class_labels)

            time_emb = self.add_class_emb(time_emb, class_emb)

        # final mp-silu for embedding

        emb = self.emb_activation(time_emb)

        # skip connections

        skips = []

        # input block

        x = self.input_block(x)

        skips.append(x)

        # down

        for encoder in self.downs:
            x = encoder(x, emb=emb)
            skips.append(x)

        # mid

        for decoder in self.mids:
            x = decoder(x, emb=emb)

        # up

        for decoder in self.ups:
            if decoder.needs_skip:
                skip = skips.pop()
                x = self.skip_mp_cat(x, skip)

            x = decoder(x, emb=emb)

        # output block

        return self.output_block(x)