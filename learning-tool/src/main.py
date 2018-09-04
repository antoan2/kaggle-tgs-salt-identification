#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import shutil
import random
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')

from skimage import io
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils.datasets import getTgsDataset
from utils.display import show_batch
from utils.submission import create_submission_file
from utils.evaluation import get_iou_vector

from models.naive import Naive

SUBMISSION = True

if __name__ == "__main__":

    dataset_train, dataloader_train = getTgsDataset('train', batch_size=16)
    dataset_validation, dataloader_validation = getTgsDataset(
        'validation', batch_size=128)

    if torch.cuda.is_available():
        model = Naive()
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print_period = 40
    n_epoches = 5

    for epoch in range(n_epoches):
        running_loss = 0.0
        t_start = time.time()
        for i, (sample) in enumerate(dataloader_train):

            output = model(sample['image'].cuda())
            loss = criterion(output, sample['mask'].cuda().long())
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_period == (
                    print_period - 1):  # print every 2000 mini-batches
                t_end = time.time()
                print('[%d, %5d] loss: %.3f elapsed time: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_period,
                       (t_end - t_start) / print_period))
                running_loss = 0.0
                t_start = time.time()
        show_batch(sample, output)

        scores = []
        for i, (sample) in enumerate(dataloader_validation):
            outputs = model(sample['image'].cuda()).detach()
            predictions = (outputs[:, 0, ...] <= outputs[:, 1, ...])
            scores.append(get_iou_vector(sample['mask'].cpu().numpy(), predictions.cpu().numpy()))
        print('Final Score', np.mean(scores))

    if SUBMISSION == True:
        # Create prediction file
        test_batch_size = 64
        dataset_test, dataloader_test = getTgsDataset(
            'test', batch_size=test_batch_size)
        create_submission_file(model, dataloader_test)
