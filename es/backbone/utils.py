from einops import pack, rearrange, repeat, unpack
from math import sqrt
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def xnor(x, y):
    return not (x ^ y)


def append(arr, el):
    arr.append(el)


def prepend(arr, el):
    arr.insert(0, el)


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return (t,) * length


def divisible_by(numer, denom):
    return (numer % denom) == 0


# in paper, they use eps 1e-4 for pixelnorm


def l2norm(t, dim=-1, eps=1e-12):
    return F.normalize(t, dim=dim, eps=eps)

def normalize_weight(weight, eps=1e-4):
    weight, ps = pack_one(weight, "o *")
    normed_weight = l2norm(weight, eps=eps)
    normed_weight = normed_weight * sqrt(weight.numel() / weight.shape[0])
    return unpack_one(normed_weight, ps, "o *")

def InvSqrtDecayLRSched(optimizer, t_ref=70000, sigma_ref=0.01):
    """
    refer to equation 67 and Table1
    """

    def inv_sqrt_decay_fn(t: int):
        return sigma_ref / sqrt(max(t / t_ref, 1.0))

    return LambdaLR(optimizer, lr_lambda=inv_sqrt_decay_fn)