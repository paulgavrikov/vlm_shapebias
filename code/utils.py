import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Callable


def str2bool(v: str) -> bool:
    return v.lower() in ("yes", "true", "t", "1", "y")


def get_key_for_value(d: dict, value: object) -> object:
    for k, v in d.items():
        if v == value:
            return k
    return None


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def process_image_as_tensor(img: Image, func: Callable) -> Image:
    x = np.asarray(img)
    x = torch.tensor(x) / 255
    x = x.permute((2, 0, 1)).unsqueeze(0)
    x = func(x)
    x = x.squeeze(0).permute((1, 2, 0)) * 255
    x = Image.fromarray(np.uint8(x))
    return x


def shuffle_image_patches(x: torch.Tensor, ps: int) -> torch.Tensor:
    # divide the batch of images into non-overlapping patches
    u = F.unfold(x, kernel_size=ps, stride=ps, padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None, ...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = F.fold(pu, x.shape[-2:], kernel_size=ps, stride=ps, padding=0)
    return f


def noise_image(x: torch.Tensor, noise_level: float) -> torch.Tensor:
    return (x + noise_level * torch.randn_like(x)).clamp(0, 1)
