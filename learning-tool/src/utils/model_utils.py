#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from tqdm import tqdm

import torch

from models.models_ensemble import ModelsEnsemble
from models.null_mask_classifiers import models as nmc_models
from models.segmentation import models as seg_models


def parse_segmentation_model(experiment_id):
    experiment_id = experiment_id.replace('seg-model-', '')
    model_type = experiment_id.split('-')[0]
    return model_type


def load_segmentation(experiment_id, no_grad=True):
    model_type = parse_segmentation_model(experiment_id)
    segmentation = ModelsEnsemble(seg_models[model_type], num_classes=2)
    segmentation.load(experiment_id)
    if no_grad == True:
        segmentation.no_grad()
    segmentation.cuda()
    return segmentation


def parse_null_mask_classifier_model(experiment_id):
    experiment_id = experiment_id.replace('nmc-model-', '')
    model_type = experiment_id.split('-')[0]
    return model_type


def load_null_mask_classifier(experiment_id, no_grad=True):
    model_type = parse_null_mask_classifier_model(experiment_id)
    null_mask_model = ModelsEnsemble(nmc_models[model_type])
    null_mask_model.load(experiment_id)
    if no_grad == True:
        null_mask_model.no_grad()
    null_mask_model.cuda()
    return null_mask_model


def apply_model(model, dataloader):
    with torch.no_grad():
        results = {}
        for samples in tqdm(dataloader):
            outputs = model(samples['image'].cuda())
            predictions = model.get_predictions(outputs).cpu().numpy()
            for sample_name, prediction in zip(samples['image_name'], predictions):
                results[sample_name] = prediction
    return results


def get_files_to_exclude(null_mask_classifier_results):
    files_to_exclude = []
    for sample_name, prediction in null_mask_classifier_results.items():
        if prediction == 0:
            files_to_exclude.append(sample_name)
    return files_to_exclude
