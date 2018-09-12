#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time

import matplotlib
matplotlib.use('Agg')

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from utils.datasets import getTgsDatasetTrainFolds, getTgsDatasetTrain
from utils.logger import Logger

from models.null_mask_classifier import NullMaskClassifier
from sklearn.metrics import accuracy_score

PRINT_PERIOD = 40
KFOLD_TRAINING = False

def train(model, dataloader, optimizer, criterion, fold, epoch, scheduler,
          logger):
    scheduler.step()
    running_loss = 0.0
    t_start = time.time()
    for batch, samples in enumerate(dataloader):
        outputs = model(samples['image'].cuda())
        loss = criterion(outputs, samples['mask_type'].cuda().long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if batch % PRINT_PERIOD == (
                PRINT_PERIOD - 1):  # print every 2000 mini-batches
            t_end = time.time()
            print('[%d, %d] loss: %.3f elapsed time: %.3f' %
                  (fold, epoch * len(dataloader) + batch + 1, running_loss / PRINT_PERIOD,
                   (t_end - t_start) / PRINT_PERIOD))
            predictions = model.get_predictions(outputs)

            logger.add_log(('train', 'loss', fold),
                           (epoch * len(dataloader) + batch, running_loss / PRINT_PERIOD))
            logger.add_log(('train', 'accuracy', fold),
                           (epoch * len(dataloader) + batch,
                            accuracy_score(samples['mask_type'], predictions)))

            t_start = time.time()
            running_loss = 0.0


def test(model, dataloader, criterion, fold, current_sample, logger):
    ground_truth = []
    predictions = []

    running_loss = 0.0
    for batch, samples in enumerate(dataloader):
        outputs = model(samples['image'].cuda()).detach()
        loss = criterion(outputs, samples['mask_type'].cuda().long())
        predictions.extend(model.get_predictions(outputs))
        ground_truth.extend(samples['mask_type'])
        running_loss += loss.item()

    logger.add_log(('validation', 'loss', fold),
                   (current_sample, running_loss / batch))
    logger.add_log(
        ('validation', 'accuracy', fold),
        (current_sample, accuracy_score(ground_truth, predictions)))


if __name__ == "__main__":

    n_epoches_kfolds = 50
    n_folds = 10
    n_epoches_train = 30
    batch_size = 16

    if KFOLD_TRAINING:
        logger = Logger()
        for fold, (dataloader_train, dataloader_validation) in enumerate(
                getTgsDatasetTrainFolds(batch_size, n_folds)):

            model = NullMaskClassifier()
            model.cuda()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.3)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(n_epoches_kfolds):
                train(model, dataloader_train, optimizer, criterion, fold, epoch,
                      scheduler, logger)
                current_sample = epoch*len(dataloader_train) + len(dataloader_train)
                test(model, dataloader_validation, criterion, fold, current_sample, logger)
        logger.plot_logs()

    else:
        logger = Logger()
        _, dataloader_train = getTgsDatasetTrain(batch_size, 1, None)
        model = NullMaskClassifier()
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.3)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(n_epoches_train):
            train(model, dataloader_train, optimizer, criterion, 1, epoch,
                  scheduler, logger)
        torch.save(model.state_dict(), 'null_mask_classifier.model')
        logger.save('null_mask_classifier.logs.pkl')
