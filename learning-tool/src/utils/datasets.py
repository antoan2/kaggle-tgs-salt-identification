#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torchvision
import torch

from dataset.tgs_salt_dataset import TgsSaltDataset
from dataset.tgs_transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RefractBorders

NUM_WORKERS = 1


def getTgsDataset(dataset, batch_size=16):
    if dataset == 'train':
        return getTgsDatasetTrainFolds(batch_size, 10)
    elif dataset == 'test':
        return getTgsDatasetTest(batch_size)


def getTgsDatasetTrainFolds(batch_size, n_folds):
    for validation_fold in range(n_folds):
        _, dataloader_train = getTgsDatasetTrain(batch_size, n_folds,
                                                 validation_fold)
        _, dataloader_validation = getTgsDatasetValidation(
            batch_size, n_folds, validation_fold)
        yield (dataloader_train, dataloader_validation)


def getTgsDatasetTrain(batch_size, n_folds, validation_fold,
                       excluded_files=[]):
    dataset = TgsSaltDataset(
        root_dir='/data',
        dataset='train',
        n_folds=n_folds,
        validation_fold=validation_fold,
        excluded_files=excluded_files,
        transform=torchvision.transforms.Compose([
            RandomHorizontalFlip(),
            # RandomVerticalFlip(),
            RefractBorders(),
            ToTensor()
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    return dataset, dataloader


def getTgsDatasetValidation(batch_size,
                            n_folds,
                            validation_fold,
                            excluded_files=[]):
    dataset = TgsSaltDataset(
        root_dir='/data',
        dataset='validation',
        n_folds=n_folds,
        validation_fold=validation_fold,
        excluded_files=excluded_files,
        transform=torchvision.transforms.Compose(
            [RefractBorders(), ToTensor()]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
    return dataset, dataloader


def getTgsDatasetTest(batch_size):
    dataset = TgsSaltDataset(
        root_dir='/data',
        dataset='test',
        transform=torchvision.transforms.Compose(
            [RefractBorders(), ToTensor()]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
    return dataset, dataloader
