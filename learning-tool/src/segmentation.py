#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import shutil
import sys
import random
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')

from skimage import io
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as torch_models

from utils.datasets import getTgsDataset, getTgsDatasetTrain, getTgsDatasetValidation
from utils.display import show_batch
from utils.submission import create_submission_file
from utils.evaluation import get_iou_vector

from models.naive import Naive
from models.linknet import LinkNet
from models.unet import UNet
from models.null_mask_classifier import NullMaskClassifier

SUBMISSION = True

if __name__ == "__main__":

    _, dataloader_train = getTgsDatasetTrain(16, 1, None)

    null_mask_model = NullMaskClassifier()
    null_mask_model.load_state_dict(torch.load('./null_mask_classifier.model'))
    null_mask_model.cuda()
    files_to_exclude = []
    for samples in dataloader_train:
        outputs = null_mask_model(samples['image'].cuda())
        predictions = null_mask_model.get_predictions(outputs)
        for sample_name, prediction in zip(samples['image_name'], predictions):
            if prediction == 0:
                files_to_exclude.append(sample_name)

    _, dataloader_train = getTgsDatasetTrain(
        16, 1, None, excluded_files=files_to_exclude)
    # 'validation', batch_size=16)

    if torch.cuda.is_available():
        # model = LinkNet(n_classes=2)
        model = UNet(n_channels=1, n_classes=2)
    model.load_state_dict(torch.load('./segmentor.model'))
    model.cuda()

    print_period = 40
    n_epoches = 40
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

    criterion = torch.nn.BCELoss()
    _, dataloader_train = getTgsDatasetTrain(
        16, 10, 0, excluded_files=files_to_exclude)
    _, dataloader_validation = getTgsDatasetValidation(
        16, 10, 0, excluded_files=files_to_exclude)

    for epoch in range(n_epoches):
        scheduler.step()
        print(optimizer.param_groups[0]['lr'])
        running_loss = 0.0
        t_start = time.time()
        for i, samples in enumerate(dataloader_train):

            outputs = model(samples['image'].cuda())
            masks_probs = F.sigmoid(outputs)
            masks_probs_flat = masks_probs.view(-1)
            true_masks = samples['mask'][:, np.newaxis, ...].cuda()
            true_masks = true_masks.repeat(1, 2, 1, 1)
            true_masks[:, 1, ...] = 1 - true_masks[:, 0, ...]

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)

            optimizer.zero_grad()
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

                predictions = model.get_predictions(outputs)

        scores = []
        for i, samples in enumerate(dataloader_validation):
            outputs = model(samples['image'].cuda()).detach()
            outputs = outputs[:, :, 14:-13, 14:-13]
            predictions = model.get_predictions(outputs)
            masks = samples['mask'].cpu().numpy()
            masks = masks[:, 14:-13, 14:-13]
            scores.append(get_iou_vector(masks, predictions.cpu().numpy()))
        print('Final Score', np.mean(scores))

    if SUBMISSION == True:
        # Create prediction file
        test_batch_size = 64
        dataset_test, dataloader_test = getTgsDataset(
            'test', batch_size=test_batch_size)
        create_submission_file(null_mask_model, model, dataloader_test)
