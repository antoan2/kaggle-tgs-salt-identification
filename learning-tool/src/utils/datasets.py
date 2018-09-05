#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torchvision
import torch

from dataset.tgs_salt_dataset import TgsSaltDataset
from dataset.tgs_transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RefractBorders

NUM_WORKERS=1

def getTgsDataset(dataset, batch_size=16):
    if dataset == 'train':
        return getTgsDatasetTrain(batch_size)
    elif dataset == 'validation':
        return getTgsDatasetValidation(batch_size)
    elif dataset == 'test':
        return getTgsDatasetTest(batch_size)


def getTgsDatasetTrain(batch_size):
    dataset = TgsSaltDataset(
        root_dir='/data',
        dataset='train',
        transform=torchvision.transforms.Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RefractBorders(),
            ToTensor()
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    return dataset, dataloader


def getTgsDatasetValidation(batch_size):
    dataset = TgsSaltDataset(
        root_dir='/data',
        dataset='validation',
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
