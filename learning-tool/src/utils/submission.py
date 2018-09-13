#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm


def create_submission_file(null_mask_model,
                           model,
                           dataloader,
                           p_file='./outputs/predictions.csv'):
    results = {}
    for i, samples in tqdm(enumerate(dataloader)):
        outputs = null_mask_model(samples['image'].cuda())
        null_mask_predictions = null_mask_model.get_predictions(outputs)

        outputs = model(samples['image'].cuda()).detach()
        outputs = outputs[:, :, 14:-13, 14:-13]

        predictions = model.get_predictions(outputs)
        for j, null_mask_prediction in enumerate(null_mask_predictions):
            if null_mask_prediction == 0:
                predictions[j, ...] = 0
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
    pixels = im.transpose().flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(rle_code):
    s = rle_code.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101).transpose()


def write_results_file(results, p_file):
    with open(p_file, 'w') as file_id:
        file_id.write('id,rle_mask\n')
        for filename, rle_code in results.items():
            file_id.write('{filename},{rle_code}\n'.format(
                filename=filename, rle_code=rle_code))
