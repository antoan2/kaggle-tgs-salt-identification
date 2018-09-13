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

            sample['image'] = sample['image'][:, ::-1]
            sample['mask'] = sample['mask'][:, ::-1]
        return sample


class RandomVerticalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            image = sample['image']
            mask = sample['mask']

            sample['image'] = sample['image'][::-1, :]
            sample['mask'] = sample['mask'][::-1, :]
        return sample


class RefractBorders(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        image = skimage.util.pad(image, ((14, 13), (14, 13)), 'reflect')
        sample['image'] = image.copy()

        if mask is not None:
            mask = skimage.util.pad(mask, ((14, 13), (14, 13)), 'reflect')
            sample['mask'] = mask.copy()
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask, mask_type = sample['image'], sample['mask'], sample[
            'mask_type']
        image_name = sample['image_name']

        image = image - 125
        image = image[np.newaxis, ...]
        image = torch.from_numpy(image).float()
        sample['image'] = image

        if mask is not None:
            mask = torch.from_numpy(mask).float()
            sample['mask'] = mask
            mask_type = torch.from_numpy(np.array(mask_type)).float()
            sample['mask_type'] = mask_type
            return sample
        else:
            sample.pop('mask')
            sample.pop('mask_type')
            return sample
