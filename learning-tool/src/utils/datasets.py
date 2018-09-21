#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torchvision
import torch

from dataset.tgs_salt_dataset import TgsSaltDataset
from dataset.tgs_transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RefractBorders

NUM_WORKERS = 1
SAMPLE_PORTION = 0.01


def getTgsDataset(dataset, batch_size=16, sample=False):
    if dataset == 'train':
        return getTgsDatasetTrainFolds(batch_size, 10, sample=sample)
    elif dataset == 'test':
        return getTgsDatasetTest(batch_size, sample=sample)


def getTgsDatasetTrainFolds(batch_size, n_folds, sample=False):
    for validation_fold in range(n_folds):
        _, dataloader_train = getTgsDatasetTrain(batch_size, n_folds,
                                                 validation_fold, sample=sample)
        _, dataloader_validation = getTgsDatasetValidation(
            batch_size, n_folds, validation_fold, sample=sample)
        yield (dataloader_train, dataloader_validation)


def getTgsDatasetTrain(batch_size,
                       n_folds,
                       validation_fold,
                       excluded_files=[],
                       sample=False):
    dataset = TgsSaltDataset(
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

    if sample:
        dataset = TgsSaltDataset(
            dataset='train',
            n_folds=n_folds,
            validation_fold=validation_fold,
            excluded_files=dataset.index_to_filename[int(SAMPLE_PORTION *
                                                         len(dataset)):],
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
                            excluded_files=[],
                            sample=False):
    dataset = TgsSaltDataset(
        dataset='validation',
        n_folds=n_folds,
        validation_fold=validation_fold,
        excluded_files=excluded_files,
        transform=torchvision.transforms.Compose(
            [RefractBorders(), ToTensor()]))

    if sample:
        dataset = TgsSaltDataset(
            dataset='validation',
            n_folds=n_folds,
            validation_fold=validation_fold,
            excluded_files=dataset.index_to_filename[int(SAMPLE_PORTION *
                                                         len(dataset)):],
            transform=torchvision.transforms.Compose(
                [RefractBorders(), ToTensor()]))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
    return dataset, dataloader


def getTgsDatasetTest(batch_size, sample=False):
    dataset = TgsSaltDataset(
        dataset='test',
        transform=torchvision.transforms.Compose(
            [RefractBorders(), ToTensor()]))

    if sample:
        dataset = TgsSaltDataset(
            dataset='test',
            excluded_files=dataset.index_to_filename[int(SAMPLE_PORTION *
                                                         len(dataset)):],
            transform=torchvision.transforms.Compose(
                [RefractBorders(), ToTensor()]))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
    return dataset, dataloader
