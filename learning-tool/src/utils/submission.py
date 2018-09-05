#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm


def create_submission_file(model, dataloader, p_file='./predictions.csv'):
    results = {}
    for i, samples in tqdm(enumerate(dataloader)):
        outputs = model(samples['image'].cuda()).detach()
        outputs = outputs[:, :, 14:-13, 14:-13]
        predictions = model.get_predictions(outputs)
        results_batch = get_encoded_results_batch(samples, predictions)
        results.update(results_batch)
    write_results_file(results, p_file)


def get_encoded_results_batch(samples, predictions):
    results = {}
    for i, (image_name, prediction) in enumerate(
            zip(samples['image_name'], predictions)):
        results[image_name] = get_encoded_results(prediction)
    return results


def get_encoded_results(prediction):
    rle_code = rle_encode(prediction.cpu().numpy())
    return rle_code


def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def write_results_file(results, p_file):
    with open(p_file, 'w') as file_id:
        file_id.write('id,rle_mask\n')
        for filename, rle_code in results.items():
            file_id.write('{filename},{rle_code}\n'.format(
                filename=filename, rle_code=rle_code))
