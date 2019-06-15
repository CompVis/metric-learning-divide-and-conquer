from __future__ import print_function
from __future__ import division

import logging
import torchvision
from torchvision import transforms
import PIL.Image
import torch


class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im


class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im


class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul_(255)
        return tensor

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor


def make(sz_resize = 256, sz_crop = 227, mean = [104, 117, 128],
        std = [1, 1, 1], rgb_to_bgr = True, is_train = True,
        intensity_scale = None):
    return transforms.Compose([
        RGBToBGR() if rgb_to_bgr else Identity(),
        transforms.RandomResizedCrop(sz_crop) if is_train else Identity(),
        transforms.Resize(sz_resize) if not is_train else Identity(),
        transforms.CenterCrop(sz_crop) if not is_train else Identity(),
        transforms.RandomHorizontalFlip() if is_train else Identity(),
        transforms.ToTensor(),
        ScaleIntensities(
            *intensity_scale) if intensity_scale is not None else Identity(),
        transforms.Normalize(
            mean=mean,
            std=std,
        )
    ])

