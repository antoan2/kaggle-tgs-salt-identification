#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import os
from skimage import io
import numpy as np
import torch
import torch.utils.data

FOLD_VALIDATION_PROPORTION = 0.1
N_FOLDS = 10


class TgsSaltDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 dataset='train',
                 transform=None,
                 excluded_files=[],
                 validation_fold=0,
                 n_folds=N_FOLDS):
        self.root_dir = root_dir
        self.dataset = dataset
        self.validation_fold = validation_fold
        self.n_folds = n_folds

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
        if not os.path.isfile(os.path.join(self.path_splits, 'train.txt')) or \
            not os.path.isfile(os.path.join(self.path_splits, 'test.txt')):
            self.create_splits()

    def check_folds(self):
        if not os.path.isfile(
                self.get_fold_filename(self.n_folds, self.validation_fold)):
            self.create_folds()

    def get_index_filenames(self):
        if self.dataset == 'test':
            return self.get_index_filenames_test()
        else:
            if self.dataset == 'validation':
                return self.get_index_filenames_validation()
            else:
                return self.get_index_filenames_train()

    def get_index_filenames_test(self):
        path_set_file = os.path.join(self.path_splits, 'test.txt')
        return [
            filename for filename in self.read_set_file(path_set_file)
            if filename not in self.excluded_files
        ]

    def get_index_filenames_validation(self):
        path_fold_file = self.get_fold_filename(self.n_folds,
                                                self.validation_fold)
        return [
            filename for filename in self.read_set_file(path_fold_file)
            if filename not in self.excluded_files
        ]

    def get_index_filenames_train(self):
        filenames = []
        for current_fold in range(self.n_folds):
            print(current_fold)
            if current_fold == self.validation_fold:
                continue
            path_fold_file = self.get_fold_filename(self.n_folds, current_fold)
            filenames.extend(self.read_set_file(path_fold_file))
        return [
            filename for filename in filenames
            if filename not in self.excluded_files
        ]

    def get_fold_filename(self, n_folds, fold):
        path_fold_file = os.path.join(
            self.root_dir, 'splits', '_'.join(
                ('folds', str(n_folds), str(fold))) + '.txt')
        return path_fold_file

    def get_folds(self):
        filenames = self.read_set_file(
            os.path.join(self.path_splits, 'train.txt'))

        random.shuffle(filenames)
        return self.chunks(filenames, int(len(filenames) / self.n_folds))

    def chunks(self, l, n):
        chunks = []
        for i in range(0, len(l), n):
            chunks.append(l[i:i + n])
        return chunks

    def __getitem__(self, index):
        filename = self.index_to_filename[index]
        path_image = os.path.join(self.path_images,
                                  '{filename}.png').format(filename=filename)
        path_mask = os.path.join(self.path_masks,
                                 '{filename}.png').format(filename=filename)

        # We made the assumption that np.std(image, axis=2) == 0
        image = io.imread(path_image)
        image = image.astype(np.float)
        image = image[:, :, 0]

        # The mask images are uselesly loaded as uint16
        if self.dataset == 'test':
            mask = None
            mask_type = None
        else:
            mask = io.imread(path_mask, as_gray=True)
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

    def create_splits(self):
        if not os.path.isdir(self.path_splits):
            os.mkdir(self.path_splits)
        train_filenames = self.list_parse_filenames(self.path_train_images)
        test_filenames = self.list_parse_filenames(self.path_test_images)
        self.write_filenames(train_filenames, 'train.txt')
        self.write_filenames(test_filenames, 'test.txt')

    def create_folds(self):
        for fold_idx, fold in enumerate(self.get_folds()):
            filenames = [filename for filename in fold]
            self.write_filenames(
                filenames, self.get_fold_filename(self.n_folds, fold_idx))

    def list_parse_filenames(self, path):
        return [os.path.splitext(filename)[0] for filename in os.listdir(path)]

    def write_filenames(self, filenames, path):
        with open(os.path.join(self.path_splits, path), 'w') as file_id:
            for filename in filenames:
                file_id.write(filename + '\n')

    def read_set_file(self, path):
        filenames = []
        with open(path) as file_id:
            for line in file_id:
                filenames.append(line.strip())
        return filenames
