#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import shutil
import sys
import random
from collections import defaultdict

from tqdm import tqdm

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
from models.models_ensemble import ModelsEnsemble

from tensorboardX import SummaryWriter

SUBMISSION = True
TIMESTAMP = str(time.time())
NULL_MASK_TIMESTAMP='1536770553.9598353'

if __name__ == "__main__":

    _, dataloader_train = getTgsDatasetTrain(16, 1, None)
    writer = SummaryWriter(log_dir='/runs/segmentor-{timestamp}'.format(
        timestamp=TIMESTAMP))

    null_mask_model = ModelsEnsemble(NullMaskClassifier)
    null_mask_model.load('1536770553.9598353')
    null_mask_model.cuda()
    files_to_exclude = []
    for samples in tqdm(dataloader_train):
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

    print_period = 100
    n_epoches = 120
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

    criterion = torch.nn.BCELoss()
    _, dataloader_train = getTgsDatasetTrain(
        16, 10, 0, excluded_files=files_to_exclude)
    _, dataloader_validation = getTgsDatasetValidation(
        16, 10, 0, excluded_files=files_to_exclude)

    for epoch in range(n_epoches):
        scheduler.step()
        running_loss = 0.0
        t_start = time.time()
        for batch, samples in enumerate(dataloader_train):

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
            current_sample = epoch*len(dataloader_train) + batch + 1
            if batch % print_period == (
                    print_period - 1):  # print every 2000 mini-batches
                t_end = time.time()
                print('[%d, %5d] loss: %.3f elapsed time: %.3f' %
                      (epoch + 1, batch + 1, running_loss / print_period,
                       (t_end - t_start) / print_period))

                outputs = outputs[:, :, 14:-13, 14:-13]
                predictions = model.get_predictions(outputs)
                masks = samples['mask'].cpu().numpy()
                masks = masks[:, 14:-13, 14:-13]

                writer.add_scalar('/data/train/loss',
                        running_loss / print_period,
                        current_sample)
                writer.add_scalar('/data/train/accuracy',
                        get_iou_vector(masks, predictions.cpu().numpy()),
                        current_sample)
                running_loss = 0.0
                t_start = time.time()


        scores = []
        running_loss = 0
        for batch, samples in enumerate(dataloader_validation):
            outputs = model(samples['image'].cuda()).detach()

            masks_probs = F.sigmoid(outputs)
            masks_probs_flat = masks_probs.view(-1)
            true_masks = samples['mask'][:, np.newaxis, ...].cuda()
            true_masks = true_masks.repeat(1, 2, 1, 1)
            true_masks[:, 1, ...] = 1 - true_masks[:, 0, ...]

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            running_loss += loss.item()

            outputs = outputs[:, :, 14:-13, 14:-13]
            predictions = model.get_predictions(outputs)
            masks = samples['mask'].cpu().numpy()
            masks = masks[:, 14:-13, 14:-13]

            current_score = get_iou_vector(masks, predictions.cpu().numpy())
            scores.append(current_score)

        writer.add_scalar('/data/test/loss',
                running_loss / (batch + 1),
                current_sample)
        writer.add_scalar('/data/test/accuracy',
                np.mean(scores),
                current_sample)

    print("Experiment Timestamp is: {TIMESTAMP}".format(TIMESTAMP))

    # Saving model
    p_model = os.path.join('/models', 'segmentor-' + TIMESTAMP)
    os.mkdir(p_model)
    torch.save(model.state_dict(), os.path.join(p_model, '0.pkl'))
    if SUBMISSION == True:
        model = UNet(n_channels=1, n_classes=2)
        model.load_state_dict(torch.load(os.path.join(p_model, '0.pkl')))
        model.cuda()
        null_mask_model = ModelsEnsemble(NullMaskClassifier)
        null_mask_model.load(NULL_MASK_TIMESTAMP)
        null_mask_model.cuda()
        # Create prediction file
        test_batch_size = 16
        dataset_test, dataloader_test = getTgsDataset(
            'test', batch_size=test_batch_size)
        create_submission_file(null_mask_model, model, dataloader_test)
