#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from skimage import io
import numpy as np
import torch

class TgsSaltDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.path_train_images = os.path.join(self.root_dir, 'train', 'images')
        self.path_train_masks = os.path.join(self.root_dir,  'train','masks')
        self.index_to_filename = [os.path.splitext(filename)[0] for filename in os.listdir(self.path_train_images)]
        self.dataset_size = len(self.index_to_filename)

        self.transform = transform

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        filename = self.index_to_filename[index]
        path_image = os.path.join(self.path_train_images, '{filename}.png').format(filename=filename)
        path_mask = os.path.join(self.path_train_masks, '{filename}.png').format(filename=filename)

        # We made the assumption that np.std(image, axis=2) == 0
        image = io.imread(path_image)
        image = image[:, :, 0]
        image = image.astype(np.float32)
        image = image[..., np.newaxis]

        # The mask images are uselesly loaded as uint16
        mask = io.imread(path_mask, as_gray=True)
        mask[mask != 0] = 1.
        mask = mask.astype(np.float32)

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample
