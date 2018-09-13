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
from models.models_ensemble import ModelsEnsemble
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter

PRINT_PERIOD = 100
KFOLD_TRAINING = True

TIMESTAMP = str(time.time())

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
        current_sample = epoch * len(dataloader) + batch + 1
        if current_sample % PRINT_PERIOD == (
                PRINT_PERIOD - 1):  # print every 2000 mini-batches
            t_end = time.time()
            print('[%d, %d, %d] loss: %.3f elapsed time: %.3f' %
                  (fold, epoch, current_sample,
                   running_loss / PRINT_PERIOD,
                   (t_end - t_start) / PRINT_PERIOD))
            predictions = model.get_predictions(outputs)

            accuracy = accuracy_score(samples['mask_type'], predictions)
            logger.add_scalar('data/train/loss',
                              running_loss / PRINT_PERIOD,
                              current_sample)
            logger.add_scalar('data/train/accuracy',
                              accuracy,
                              current_sample)

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

    logger.add_scalar('data/test/loss',
                      running_loss / (batch + 1),
                      current_sample)
    logger.add_scalar('data/test/accuracy',
                      accuracy_score(ground_truth, predictions),
                      current_sample)


if __name__ == "__main__":

    n_epoches_kfolds = 30
    n_folds = 4
    n_epoches_train = 30
    batch_size = 16

    logger = Logger()
    models_ensemble = ModelsEnsemble(NullMaskClassifier)
    for fold, (dataloader_train, dataloader_validation) in enumerate(
            getTgsDatasetTrainFolds(batch_size, n_folds)):
        writer = SummaryWriter(log_dir='/runs/null-mask-{timestamp}-fold-{fold}'.format(
            timestamp=TIMESTAMP,
            fold=fold))

        model = NullMaskClassifier()
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.3)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(n_epoches_kfolds):
            train(model, dataloader_train, optimizer, criterion, fold,
                  epoch, scheduler, writer)
            current_sample = epoch * len(dataloader_train) + len(
                dataloader_train)
            test(model, dataloader_validation, criterion, fold,
                 current_sample, writer)
        # logger.plot_logs()
        writer.close()
        models_ensemble.add_model(model)
    models_ensemble.save(TIMESTAMP)

    print("Experiment Timestamp is: {TIMESTAMP}".format(TIMESTAMP))
