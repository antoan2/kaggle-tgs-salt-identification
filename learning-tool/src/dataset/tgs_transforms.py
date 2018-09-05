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
            image_name = sample['image_name']
            mask = sample['mask']

            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

            return {'image_name': image_name, 'image': image, 'mask': mask}
        else:
            return sample


class RandomVerticalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            image = sample['image']
            image_name = sample['image_name']
            mask = sample['mask']

            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

            return {'image_name': image_name, 'image': image, 'mask': mask}
        else:
            return sample

class RefractBorders(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image_name = sample['image_name']

        image = skimage.util.pad(image, ((14, 13), (14, 13)), 'reflect')

        if mask is None:
            return {'image_name': image_name, 'image': image, 'mask': None}
        else:
            mask = skimage.util.pad(mask, ((14, 13), (14, 13)), 'reflect')
            return {
                'image_name': image_name,
                'image': image,
                'mask': mask
            }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image_name = sample['image_name']

        # image /= 255.
        image -= 125
        image = image[np.newaxis, ...]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if mask is None:
            return {'image_name': image_name, 'image': torch.from_numpy(image).float()}
        else:
            return {
                'image_name': image_name,
                'image': torch.from_numpy(image).float(),
                'mask': torch.from_numpy(mask).float()
            }
