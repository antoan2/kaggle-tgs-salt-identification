#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import argparse
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from utils.datasets import getTgsDatasetValidation, getTgsDatasetTrain

from models.null_mask_classifiers import models as nmc_models
from models.models_ensemble import ModelsEnsemble
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter

from lr_schedulers.cosinus_annealing_lr_with_restart import CosineAnnealingLRWithRestart

PRINT_PERIOD = 50


def train(model, dataloader, optimizer, criterion, epoch, scheduler, writer):
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
            current_lr = optimizer.param_groups[0]['lr']
            print('[%d, %d] loss: %.3f elapsed time: %.3f lr: %.5f' %
                  (epoch, current_sample, running_loss / PRINT_PERIOD,
                   (t_end - t_start) / PRINT_PERIOD, current_lr))
            predictions = model.get_predictions(outputs)

            accuracy = accuracy_score(samples['mask_type'], predictions)
            writer.add_scalar('data/train/loss', running_loss / PRINT_PERIOD,
                              current_sample)
            writer.add_scalar('data/train/accuracy', accuracy, current_sample)
            writer.add_scalar('/data/train/learning_rate', current_lr,
                              current_sample)

            t_start = time.time()
            running_loss = 0.0


def test(model, dataloader, criterion, current_sample, writer):
    ground_truth = []
    predictions = []

    running_loss = 0.0
    for batch, samples in enumerate(dataloader):
        outputs = model(samples['image'].cuda()).detach()
        loss = criterion(outputs, samples['mask_type'].cuda().long())
        predictions.extend(model.get_predictions(outputs))
        ground_truth.extend(samples['mask_type'])
        running_loss += loss.item()

    writer.add_scalar('data/test/loss', running_loss / (batch + 1),
                      current_sample)
    writer.add_scalar('data/test/accuracy',
                      accuracy_score(ground_truth, predictions),
                      current_sample)


def test_final_model(model, dataloader, criterion):
    ground_truth = []
    predictions = []

    running_loss = 0.0
    for batch, samples in enumerate(dataloader):
        outputs = model(samples['image'].cuda()).detach()
        loss = criterion(outputs, samples['mask_type'].cuda().long())
        predictions.extend(model.get_predictions(outputs))
        ground_truth.extend(samples['mask_type'])
        running_loss += loss.item()
    return running_loss / (batch + 1), accuracy_score(ground_truth,
                                                      predictions)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


def main(args):
    experiment_id = create_experiment_id(args)

    print('Experiment running is: {experiment_id}'.format(
        experiment_id=experiment_id))

    writer = SummaryWriter(log_dir='/runs/{experiment_id}'.format(
        experiment_id=experiment_id))

    model_type = nmc_models[args.model]
    models_ensemble = ModelsEnsemble(model_type)
    model = model_type()
    model.cuda()

    _, dataloader_train = getTgsDatasetTrain(args.batch_size, 10, 0)
    _, dataloader_validation = getTgsDatasetValidation(args.batch_size, 10, 0)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLRWithRestart(
        optimizer, T_max=args.t_max, eta_min=args.min_learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.n_epoches):

        snapshot_to_be_saved = scheduler.step()

        train(model, dataloader_train, optimizer, criterion, epoch, scheduler,
              writer)
        current_sample = epoch * len(dataloader_train) + len(dataloader_train)
        test(model, dataloader_validation, criterion, current_sample, writer)

        if snapshot_to_be_saved:
            print('saving model')
            models_ensemble.add_model(model)
    writer.close()
    models_ensemble.save(experiment_id)

    final_loss, final_accuracy = test_final_model(model, dataloader_validation,
                                                  criterion)
    print('Experiment finishing is: {experiment_id}'.format(
        experiment_id=experiment_id))
    print('Final loss: {final_loss}'.format(final_loss=final_loss))
    print('Final accuracy: {final_accuracy}'.format(
        final_accuracy=final_accuracy))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default='Resnet18',
        type=check_model,
        help='The model to use')
    parser.add_argument(
        '--n_epoches',
        default=30,
        type=int,
        help='number of epoches to train the model')
    parser.add_argument(
        '--t_max',
        default=10,
        type=int,
        help='period to restart the learning rate')
    parser.add_argument(
        '--learning_rate',
        default=0.01,
        type=float,
        help='starting learning rate')
    parser.add_argument(
        '--min_learning_rate',
        default=0.0001,
        type=float,
        help='min learning rate')
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
    id_parts = ['nmc']
    id_parts.extend(('model', args.model))
    id_parts.extend(('epoches', args.n_epoches))
    id_parts.extend(('t_max', args.t_max))
    id_parts.extend(('lr', args.learning_rate))
    id_parts.extend(('min_lr', args.min_learning_rate))
    id_parts.extend(('batch_size', args.batch_size))
    id_parts.extend(('timestamp', timestamp))
    return '-'.join([str(id_part) for id_part in id_parts])


if __name__ == "__main__":
    args = parse_args()
    main(args)
