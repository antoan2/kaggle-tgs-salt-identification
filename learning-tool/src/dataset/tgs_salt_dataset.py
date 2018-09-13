#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import os
from skimage import io
import numpy as np
import torch
import torch.utils.data

P_ORIGINAL_DATA = '/original_data'
P_DATA = '/data'


class TgsSaltDataset(torch.utils.data.Dataset):
    def __init__(self,
                 p_original_data=P_ORIGINAL_DATA,
                 p_data=P_DATA,
                 dataset='train',
                 transform=None,
                 validation_fold=0,
                 n_folds=5,
                 excluded_files=[]):
        self.p_original_data = p_original_data
        self.p_data = p_data
        self.dataset = dataset
        self.validation_fold = validation_fold
        self.n_folds = n_folds

        # Set all paths as a dict
        self.p_splits = os.path.join(self.p_data, 'splits')
        self.p_test_images = os.path.join(self.p_original_data, 'test',
                                          'images')
        self.p_train_images = os.path.join(self.p_original_data, 'train',
                                           'images')
        self.p_train_masks = os.path.join(self.p_original_data, 'train',
                                          'masks')

        if dataset == 'test':
            self.p_images = self.p_test_images
            self.p_masks = None
        else:
            self.p_images = self.p_train_images
            self.p_masks = self.p_train_masks

        self.check_splits_folds()
        self.excluded_files = excluded_files
        self.index_to_filename = self.get_index_filenames()

        self.dataset_size = len(self.index_to_filename)

        self.transform = transform

    def __len__(self):
        return self.dataset_size

    def check_splits_folds(self):
        self.check_splits()
        self.check_folds()

    def check_splits(self):
        if not os.path.isfile(os.path.join(self.p_splits, 'train.txt')) or \
            not os.path.isfile(os.path.join(self.p_splits, 'test.txt')):
            self.create_splits()

    def create_splits(self):
        if not os.path.isdir(self.p_splits):
            os.mkdir(self.p_splits)
        train_filenames = self.list_folder_filenames(self.p_train_images)
        test_filenames = self.list_folder_filenames(self.p_test_images)
        self.write_filenames(train_filenames, 'train.txt')
        self.write_filenames(test_filenames, 'test.txt')

    def list_folder_filenames(self, path):
        return [os.path.splitext(filename)[0] for filename in os.listdir(path)]

    def write_filenames(self, filenames, path):
        with open(os.path.join(self.p_splits, path), 'w') as file_id:
            for filename in filenames:
                file_id.write(filename + '\n')

    def check_folds(self):
        if not os.path.isfile(
                self.get_fold_filename(self.n_folds, self.validation_fold)):
            self.create_folds()

    def get_fold_filename(self, n_folds, fold):
        p_fold_file = os.path.join(
            self.p_splits, '_'.join(
                ('folds', str(n_folds), str(fold))) + '.txt')
        return p_fold_file

    def create_folds(self):
        for fold_idx, fold in enumerate(self.get_folds()):
            filenames = [filename for filename in fold]
            self.write_filenames(
                filenames, self.get_fold_filename(self.n_folds, fold_idx))

    def get_folds(self):
        filenames = self.read_dataset_file(
            os.path.join(self.p_splits, 'train.txt'))

        random.shuffle(filenames)
        return self.chunks(filenames, int(len(filenames) / self.n_folds))

    def chunks(self, l, n):
        chunks = []
        for i in range(0, len(l), n):
            chunks.append(l[i:i + n])
        return chunks

    def read_dataset_file(self, path):
        filenames = []
        with open(path) as file_id:
            for line in file_id:
                filenames.append(line.strip())
        return filenames

    def get_index_filenames(self):
        if self.dataset == 'test':
            return self.get_index_filenames_test()
        else:
            if self.dataset == 'validation':
                return self.get_index_filenames_validation()
            else:
                return self.get_index_filenames_train()

    def get_index_filenames_test(self):
        p_dataset_file = os.path.join(self.p_splits, 'test.txt')
        return [
            filename for filename in self.read_dataset_file(p_dataset_file)
            if filename not in self.excluded_files
        ]

    def get_index_filenames_validation(self):
        p_fold_file = self.get_fold_filename(self.n_folds,
                                             self.validation_fold)
        return [
            filename for filename in self.read_dataset_file(p_fold_file)
            if filename not in self.excluded_files
        ]

    def get_index_filenames_train(self):
        filenames = []
        for current_fold in range(self.n_folds):
            if current_fold == self.validation_fold:
                continue
            p_fold_file = self.get_fold_filename(self.n_folds, current_fold)
            filenames.extend(self.read_dataset_file(p_fold_file))
        return [
            filename for filename in filenames
            if filename not in self.excluded_files
        ]

    def __getitem__(self, index):
        filename = self.index_to_filename[index]
        p_image = os.path.join(self.p_images,
                               '{filename}.png').format(filename=filename)

        # We made the assumption that np.std(image, axis=2) == 0
        image = io.imread(p_image)
        image = image.astype(np.float)
        image = image[:, :, 0]

        # The mask images are uselesly loaded as uint16
        if self.dataset == 'test':
            mask = None
            mask_type = None
        else:
            p_mask = os.path.join(self.p_masks,
                                  '{filename}.png').format(filename=filename)
            mask = io.imread(p_mask, as_gray=True)
            mask = mask.astype(np.float)
            mask[mask != 0] = 1.
            if np.count_nonzero(mask) == 0:
                mask_type = 0
            else:
                mask_type = 1

        sample = {
            'image_name': filename,
            'image': image,
            'mask': mask,
            'mask_type': mask_type
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
