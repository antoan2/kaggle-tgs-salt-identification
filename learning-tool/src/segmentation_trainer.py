#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import shutil
import sys
import random
from collections import defaultdict
import argparse

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

from models.segmentation import models as seg_models
from models.null_mask_classifiers import models as nmc_models
from models.models_ensemble import ModelsEnsemble
from losses.lovasz_losses import lovasz_softmax

from tensorboardX import SummaryWriter

SUBMISSION = True
PRINT_PERIOD = 10
TIMESTAMP = str(time.time())
NULL_MASK_TIMESTAMP = '1536770553.9598353'



def parse_null_mask_classifier_model(experiment_id):
    experiment_id = experiment_id.replace('null_mask_classifier-model-', '')
    model_type = experiment_id.split('-')[0]
    return model_type


def load_null_mask_classifier(experiment_id):
    model_type = parse_null_mask_classifier_model(experiment_id)
    null_mask_model = ModelsEnsemble(nmc_models[model_type])
    null_mask_model.load(experiment_id)
    null_mask_model.cuda()
    return null_mask_model

def get_files_to_exclude(null_mask_classifier, dataloader):
    files_to_exclude = []
    for samples in tqdm(dataloader):
        outputs = null_mask_classifier(samples['image'].cuda())
        predictions = null_mask_classifier.get_predictions(outputs)
        for sample_name, prediction in zip(samples['image_name'], predictions):
            if prediction == 0:
                files_to_exclude.append(sample_name)
    return files_to_exclude

def train(model, dataloader, optimizer, criterion, epoch, scheduler, writer):
    scheduler.step()
    running_loss = 0.0
    t_start = time.time()
    for batch, samples in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(samples['image'].cuda())
        output_probabilities = F.softmax(outputs, dim=1)

        loss = criterion(output_probabilities, samples['mask'].cuda(), ignore=255)

        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        current_sample = epoch * len(dataloader) + batch + 1
        if batch % PRINT_PERIOD == (
                PRINT_PERIOD - 1):  # print every 2000 mini-batches
            t_end = time.time()
            print('[%d, %5d] loss: %.3f elapsed time: %.3f' %
                  (epoch + 1, batch + 1, running_loss / PRINT_PERIOD,
                   (t_end - t_start) / PRINT_PERIOD))

            outputs = outputs[:, :, 14:-13, 14:-13]
            predictions = model.get_predictions(outputs)
            masks = samples['mask'].cpu().numpy()
            masks = masks[:, 14:-13, 14:-13]

            writer.add_scalar('/data/train/loss',
                              running_loss / PRINT_PERIOD, current_sample)
            writer.add_scalar(
                '/data/train/accuracy',
                get_iou_vector(masks,
                               predictions.cpu().numpy()), current_sample)
            running_loss = 0.0
            t_start = time.time()

def test(model, dataloader, criterion, current_sample, writer):
    scores = []
    running_loss = 0
    for batch, samples in enumerate(dataloader):
        outputs = model(samples['image'].cuda()).detach()
        output_probabilities = F.softmax(outputs, dim=1)

        loss = criterion(output_probabilities, samples['mask'].cuda())
        running_loss += loss.item()

        outputs = outputs[:, :, 14:-13, 14:-13]
        predictions = model.get_predictions(outputs)
        masks = samples['mask'].cpu().numpy()
        masks = masks[:, 14:-13, 14:-13]

        current_score = get_iou_vector(masks, predictions.cpu().numpy())
        scores.append(current_score)

    writer.add_scalar('/data/test/loss', running_loss / (batch + 1),
                      current_sample)
    writer.add_scalar('/data/test/accuracy', np.mean(scores),
                      current_sample)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--null_mask_classifier',
        type=str,
        default=None,
        help='The null_mask_classifier id')
    parser.add_argument(
        '--submission',
        action='store_true',
        help='If we create a submission file')
    parser.add_argument(
        '--model', default='UNet', type=check_model, help='The model to use')
    parser.add_argument(
        '--n_epoches',
        default=30,
        type=int,
        help='number of epoches to train the model')
    parser.add_argument(
        '--learning_rate',
        default=0.01,
        type=float,
        help='starting learning rate')
    parser.add_argument(
        '--batch_size', default=16, type=int, help='size of the batch')
    return parser.parse_args()

def check_model(value):
    if value not in seg_models:
        raise argparse.ArgumentTypeError(
            '%s is not a valid model type. Valid model types are %s' %
            (value, str(seg_models.keys())))
    return value

def create_experiment_id(args):
    timestamp = str(time.time())
    id_parts = ['segmentation']
    id_parts.extend(('model', args.model))
    id_parts.extend(('epoches', args.n_epoches))
    id_parts.extend(('lr', args.learning_rate))
    id_parts.extend(('batch_size', args.batch_size))
    id_parts.extend(('timestamp', timestamp))
    return '-'.join([str(id_part) for id_part in id_parts])

def main(args):
    experiment_id = create_experiment_id(args)
    print('Experiment running is: {experiment_id}'.format(experiment_id=experiment_id))

    if args.null_mask_classifier is not None:
        null_mask_classifier = load_null_mask_classifier(args.null_mask_classifier)
        _, dataloader_train = getTgsDatasetTrain(args.batch_size, 1, None)
        files_to_exclude = get_files_to_exclude(null_mask_classifier, dataloader_train)
    else:
        null_mask_classifier = None
        files_to_exclude = []

    writer = SummaryWriter(log_dir=os.path.join('/runs', experiment_id))

    _, dataloader_train = getTgsDatasetTrain(
        args.batch_size, 1, None, excluded_files=files_to_exclude)

    if torch.cuda.is_available():
        model = seg_models[args.model](num_classes=2)
    model.cuda()

    n_epoches = args.n_epoches
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    # criterion = torch.nn.BCELoss()
    criterion = lovasz_softmax

    _, dataloader_train = getTgsDatasetTrain(
        args.batch_size, 10, 0, excluded_files=files_to_exclude)
    _, dataloader_validation = getTgsDatasetValidation(
        args.batch_size, 10, 0, excluded_files=files_to_exclude)

    for epoch in range(n_epoches):
        train(model, dataloader_train, optimizer,criterion, epoch, scheduler, writer)
        current_sample = epoch * len(dataloader_train) + len(
            dataloader_train)
        test(model, dataloader_validation, criterion, current_sample, writer)

    # Saving model
    p_model = os.path.join('/models', experiment_id)
    os.mkdir(p_model)
    torch.save(model.state_dict(), os.path.join(p_model, '0.pkl'))

    if args.submission == True:
        model = seg_models[args.model](num_classes=2)
        model.load_state_dict(torch.load(os.path.join(p_model, '0.pkl')))
        model.cuda()
        # Create prediction file
        test_batch_size = args.batch_size
        dataset_test, dataloader_test = getTgsDataset(
            'test', batch_size=test_batch_size)
        create_submission_file(null_mask_classifier, model, dataloader_test,
                p_file=os.path.join('./outputs', experiment_id + '.csv'))

    print('Experiment finishing is: {experiment_id}'.format(experiment_id=experiment_id))

if __name__ == "__main__":

    args = parse_args()
    main(args)
