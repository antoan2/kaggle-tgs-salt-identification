#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
import skimage


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            image = sample['image']
            mask = sample['mask']

            sample['image'] = np.flip(image, axis=0).copy()
            sample['mask'] = np.flip(mask, axis=0).copy()
        return sample


class RandomVerticalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            image = sample['image']
            mask = sample['mask']

            sample['image'] = np.flip(image, axis=1).copy()
            sample['mask'] = np.flip(mask, axis=1).copy()
        return sample


class RefractBorders(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        image = skimage.util.pad(image, ((14, 13), (14, 13)), 'reflect')
        sample['image'] = image

        if mask is not None:
            mask = skimage.util.pad(mask, ((14, 13), (14, 13)), 'reflect')
            sample['mask'] = mask
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image_name = sample['image_name']

        image = image - 125
        image = image[np.newaxis, ...]
        image = torch.from_numpy(image).float()
        sample['image'] = image

        if mask is not None:
            mask = torch.from_numpy(mask).float()
            sample['mask'] = mask

        return sample
