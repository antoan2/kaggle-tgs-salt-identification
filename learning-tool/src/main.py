#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import shutil
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage import io
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from dataset.tgs_salt_dataset import TgsSaltDataset
from dataset.tgs_transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip

# from models.fcn import FCN
from models.first import First


def show_batch(sample_batched, predictions=None):

    fig = plt.figure()
    if predictions is None:
        fig.add_subplot(211)
    else:
        fig.add_subplot(311)
    grid_images = torchvision.utils.make_grid(sample_batched['image'], normalize=True)
    plt.imshow(grid_images.numpy().transpose((1, 2, 0)))

    if predictions is None:
        fig.add_subplot(212)
    else:
        fig.add_subplot(312)
    grid_masks = torchvision.utils.make_grid(sample_batched['mask'][:, np.newaxis, ...])
    grid_masks = grid_masks.numpy()
    plt.imshow(grid_masks.transpose((1, 2, 0)))

    if predictions is not None:
        fig.add_subplot(313)
        predictions = predictions.detach()
        predictions = (predictions[:, 0, ...] <= predictions[:, 1, ...])
        grid_predictions = torchvision.utils.make_grid(predictions[:, np.newaxis, ...])
        grid_predictions = grid_predictions.cpu().numpy().astype(np.float)
        plt.imshow(grid_predictions.transpose((1, 2, 0)))
    plt.savefig('./test.png')
    plt.close()


def check_same_depth():
    # Getting depths
    with open('/data/depths.csv') as file_id:
        file_id.readline()
        depths = defaultdict(list)
        for line in file_id:
            filename, depth = line.strip().split(',')
            depths[int(depth)].append(filename)
    # Getting train examples
    train_filenames = [filename[:-4] for filename in os.listdir('/data/train/images')]
    test_filenames = [filename[:-4] for filename in os.listdir('/data/test/images')]
    os.mkdir('temp_folder')
    for depth, filenames in depths.items():
        os.mkdir(os.path.join('temp_folder', str(depth)))
        for filename in filenames:
            if filename in train_filenames:
                shutil.copyfile('/data/train/images/%s.png' % filename, os.path.join('temp_folder', str(depth), filename + '.png'))
            else:
                shutil.copyfile('/data/test/images/%s.png' % filename, os.path.join('temp_folder', str(depth), filename + '.png'))



    return depths, train_filenames



if __name__ == "__main__":
    tgs_dataset = TgsSaltDataset(
        root_dir='/data',
        transform=torchvision.transforms.Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor()]))
    dataloader = torch.utils.data.DataLoader(
        tgs_dataset, batch_size=16, shuffle=True, num_workers=1)

    if torch.cuda.is_available():
        model = First()
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    print_period = 40
    n_epoches = 20


    for epoch in range(n_epoches):
        running_loss = 0.0
        for i, (sample) in enumerate(dataloader):

            output = model(sample['image'].cuda())
            loss = criterion(output, sample['mask'].cuda().long())
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_period == (print_period - 1):    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_period))
                running_loss = 0.0
        show_batch(sample, output)
