#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import os
from skimage import io
import numpy as np
import torch


class TgsSaltDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, dataset='train', transform=None):
        self.root_dir = root_dir
        self.dataset = dataset
        # Set all paths as a dict
        self.path_splits = os.path.join(self.root_dir, 'splits')
        self.path_train_images = os.path.join(self.root_dir, 'train', 'images')
        self.path_train_masks = os.path.join(self.root_dir, 'train', 'masks')
        self.path_test_images = os.path.join(self.root_dir, 'test', 'images')
        self.path_test_masks = os.path.join(self.root_dir, 'test', 'masks')

        if dataset == 'test':
            self.path_images = self.path_test_images
            self.path_masks = self.path_test_masks
        else:
            self.path_images = self.path_train_images
            self.path_masks = self.path_train_masks

        try:
            self.index_to_filename = self.get_index_filenames()
        except:
            self.create_splits()
            self.index_to_filename = self.get_index_filenames()
        self.dataset_size = len(self.index_to_filename)

        self.transform = transform

    def __len__(self):
        return self.dataset_size

    def get_index_filenames(self):
        filenames = []
        with open(
                os.path.join(self.root_dir, 'splits',
                             self.dataset + '.txt')) as file_id:
            for line in file_id:
                filenames.append(line.strip())
        return filenames

    def __getitem__(self, index):
        filename = self.index_to_filename[index]
        path_image = os.path.join(self.path_images,
                                  '{filename}.png').format(filename=filename)
        path_mask = os.path.join(self.path_masks,
                                 '{filename}.png').format(filename=filename)

        # We made the assumption that np.std(image, axis=2) == 0
        image = io.imread(path_image)
        image = image[:, :, 0]
        image = image.astype(np.float32)
        image = image[..., np.newaxis]

        # The mask images are uselesly loaded as uint16
        if self.dataset == 'test':
            mask = None
        else:
            mask = io.imread(path_mask, as_gray=True)
            mask[mask != 0] = 1.
            mask = mask.astype(np.float32)

        sample = {'image_name': filename, 'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def create_splits(self):
        if not os.path.isdir(self.path_splits):
            os.mkdir(self.path_splits)

        filenames = [
            os.path.splitext(filename)[0]
            for filename in os.listdir(self.path_train_images)
        ]
        n_test_samples = int(len(filenames) / 10.)
        test_set = random.sample(filenames, n_test_samples)
        train_set = set(filenames).difference(test_set)
        print(len(test_set), len(train_set))

        with open(os.path.join(self.path_splits, 'validation.txt'),
                  'w') as file_id:
            for filename in test_set:
                file_id.write(filename + '\n')
        with open(os.path.join(self.path_splits, 'train.txt'), 'w') as file_id:
            for filename in train_set:
                file_id.write(filename + '\n')

        filenames = [
            os.path.splitext(filename)[0]
            for filename in os.listdir(self.path_test_images)
        ]
        with open(os.path.join(self.path_splits, 'test.txt'), 'w') as file_id:
            for filename in filenames:
                file_id.write(filename + '\n')
