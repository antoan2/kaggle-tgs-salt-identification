#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm


def create_null_mask_classifier_predictions_file(files, files_to_exclude,
                                                 p_file):
    results = {}
    for filename in files:
        if filename in files_to_exclude:
            results[filename] = ''
        else:
            results[filename] = '1 1'
    write_results_file(results, p_file)

def get_encoded_results(results, files_to_exclude=[]):
    encoded_results = {}
    for sample_name, prediction in tqdm(results.items()):
        if sample_name in files_to_exclude:
            encoded_results[sample_name] = ''
        else:
            encoded_results[sample_name] = get_encoded_result(prediction)
    return encoded_results

def get_encoded_result(prediction):
    rle_code = rle_encode(prediction)
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
