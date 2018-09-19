#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import argparse

import matplotlib
matplotlib.use('Agg')

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from utils.datasets import getTgsDatasetTrainFolds, getTgsDatasetTrain
from utils.logger import Logger

from models.null_mask_classifiers import models as nmc_models
from models.models_ensemble import ModelsEnsemble
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter

PRINT_PERIOD = 100


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
                  (fold, epoch, current_sample, running_loss / PRINT_PERIOD,
                   (t_end - t_start) / PRINT_PERIOD))
            predictions = model.get_predictions(outputs)

            accuracy = accuracy_score(samples['mask_type'], predictions)
            logger.add_scalar('data/train/loss', running_loss / PRINT_PERIOD,
                              current_sample)
            logger.add_scalar('data/train/accuracy', accuracy, current_sample)

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

    logger.add_scalar('data/test/loss', running_loss / (batch + 1),
                      current_sample)
    logger.add_scalar('data/test/accuracy',
                      accuracy_score(ground_truth, predictions),
                      current_sample)


def main(args):
    experiment_id = create_experiment_id(args)

    print('Experiment running is: {experiment_id}'.format(experiment_id=experiment_id))

    logger = Logger()
    model_type = nmc_models[args.model]
    models_ensemble = ModelsEnsemble(model_type)
    for fold, (dataloader_train, dataloader_validation) in enumerate(
            getTgsDatasetTrainFolds(args.batch_size, args.n_folds)):
        writer = SummaryWriter(
            log_dir='/runs/{experiment_id}-fold-{fold}'.format(
                experiment_id=experiment_id, fold=fold))

        model = model_type()
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.learnin_rate)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.3)

        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(args.n_epoches):
            train(model, dataloader_train, optimizer, criterion, fold, epoch,
                  scheduler, writer)
            current_sample = epoch * len(dataloader_train) + len(
                dataloader_train)
            test(model, dataloader_validation, criterion, fold, current_sample,
                 writer)
        # logger.plot_logs()
        writer.close()
        models_ensemble.add_model(model)
    models_ensemble.save(experiment_id)
    print('Experiment finishing is: {experiment_id}'.format(experiment_id=experiment_id))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default='Resnet18',
        type=check_model,
        help='The model to use')
    parser.add_argument(
        '--n_folds',
        default=4,
        type=int,
        help='number of folds used to train the model')
    parser.add_argument(
        '--n_epoches',
        default=30,
        type=int,
        help='number of epoches to train the model')
    parser.add_argument(
        '--learnin_rate',
        default=0.01,
        type=float,
        help='starting learning rate')
    parser.add_argument(
        '--batch_size', default=16, type=int, help='size of the batch')
    return parser.parse_args()

def check_model(value):
    if value not in nmc_models:
        raise argparse.ArgumentTypeError(
            '%s is not a valid model type. Valid model types are %s' %
            (value, str(nmc_models.keys())))
    return value


def create_experiment_id(args):
    timestamp = str(time.time())
    id_parts = ['null_mask_classifier']
    id_parts.extend(('model', args.model))
    id_parts.extend(('n_folds', args.n_folds))
    id_parts.extend(('epoches', args.n_epoches))
    id_parts.extend(('lr', args.learnin_rate))
    id_parts.extend(('batch_size', args.batch_size))
    id_parts.extend(('timestamp', timestamp))
    return '-'.join([str(id_part) for id_part in id_parts])

if __name__ == "__main__":
    args = parse_args()
    main(args)
