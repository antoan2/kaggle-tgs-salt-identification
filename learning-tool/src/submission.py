#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script will load the specified models, generate the submission files and a script with the kaggle submissions commands.
If a nmc is provided, it will generate a submissions files with only empty results excepted where the nmc predict a non empty mask (as suggested in this discussion: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65933
If a seg model is provided, it will generate the submission file with rle codes and if a nmc is also provided the submission files without the "empty mask" predicted images.
It can also generate postprocessed output files.
"""
import os
import argparse

from utils.datasets import getTgsDataset
from utils.model_utils import load_null_mask_classifier, load_segmentation, apply_model, get_files_to_exclude
from utils.submission import create_null_mask_classifier_predictions_file, get_encoded_results, write_results_file
from postprocessing import post_process

SAMPLE = False
P_OUTPUTS = '/outputs/'
P_OUTPUTS_REAL_WORD = os.environ["P_OUTPUTS"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-nmc',
        '--null_mask_classifier',
        type=str,
        default=None,
        help='The null_mask_classifier model to use')
    parser.add_argument(
        '-seg',
        '--segmentation',
        type=str,
        default=None,
        help='The segmentation model to use')
    parser.add_argument(
        '-p',
        '--post_processing',
        default=False,
        action="store_true",
        help='Using post_processing')
    return parser.parse_args()


def create_submission_script(generated_files):
    kaggle_cmd_template = 'kaggle competitions submit -c tgs-salt-identification-challenge \\\n -f "{p_outputs}" \\\n -m "{message}"'
    kaggle_cmds = []
    for p_outputs in generated_files:
        message = p_outputs.replace(P_OUTPUTS, '')
        p_outputs_real_world = os.path.join(P_OUTPUTS_REAL_WORD, message)
        kaggle_cmds.append(kaggle_cmd_template.format(p_outputs=p_outputs_real_world, message=message))
    with open('./submission_script.sh', 'w') as file_id:
        file_id.write('set -ex\n')
        for kaggle_cmd in kaggle_cmds:
            file_id.write(kaggle_cmd)
            file_id.write('\n\n')

def exclude_files(results, files_to_exclude):
    filtered_results = {}
    for sample_name, result in results.items():
        if sample_name in files_to_exclude:
            filtered_results[sample_name] = ''
        else:
            filtered_results[sample_name] = result
    return filtered_results

def crop_results(results):
    return {
        sample_name: prediction[14:-13, 14:-13]
        for sample_name, prediction in results.items()
    }


def create_output_post_processed(rle_codes_post_processed, p_output_file):
    p_output_post_processed = get_post_process_filename(p_output_file)
    write_results_file(rle_codes_post_processed, p_output_post_processed)
    return p_output_post_processed

def get_post_process_filename(p_output_file):
    return os.path.splitext(p_output_file)[0] + '_post_processing.csv'

def main(args):
    dataset_test, dataloader_test = getTgsDataset('test', sample=SAMPLE)
    generated_files = []

    if args.null_mask_classifier is not None:
        null_mask_classifier = load_null_mask_classifier(
            args.null_mask_classifier)
        null_mask_classifier_results = apply_model(null_mask_classifier,
                                                   dataloader_test)
        files_to_exclude = get_files_to_exclude(null_mask_classifier_results)

        p_output_nmc = os.path.join(P_OUTPUTS,
                                     args.null_mask_classifier + '.csv')
        create_null_mask_classifier_predictions_file(
            dataset_test.index_to_filename, files_to_exclude, p_output_nmc)
        generated_files.append(p_output_nmc)

    if args.segmentation is not None:
        segmentation = load_segmentation(args.segmentation)
        results = apply_model(segmentation, dataloader_test)
        cropped_results = crop_results(results)

        seg_rle_codes = get_encoded_results(cropped_results)

        p_output_seg = os.path.join(P_OUTPUTS, 'nmc_None_' + args.segmentation + '.csv')
        write_results_file(seg_rle_codes, p_output_seg)
        generated_files.append(p_output_seg)



        if args.null_mask_classifier is not None:
            nmc_seg_rle_codes = get_encoded_results(cropped_results, files_to_exclude)

            p_output_nmc_seg = os.path.join(P_OUTPUTS, args.null_mask_classifier + '_' + args.segmentation + '.csv')
            write_results_file(nmc_seg_rle_codes, p_output_nmc_seg)
            generated_files.append(p_output_nmc_seg)

        if args.post_processing:
            seg_rle_codes_post_processed = post_process(seg_rle_codes)

            p_output_seg_post_processed = create_output_post_processed(seg_rle_codes_post_processed, p_output_seg)
            generated_files.append(p_output_seg_post_processed)

            if args.null_mask_classifier:
                nmc_seg_rle_codes_post_processed = exclude_files(seg_rle_codes_post_processed, files_to_exclude)

                p_output_nmc_seg_post_processed = create_output_post_processed(nmc_seg_rle_codes_post_processed, p_output_nmc_seg)
                generated_files.append(p_output_nmc_seg_post_processed)

    create_submission_script(generated_files)
    print("Generated files:")
    for generated_file in generated_files:
        print("- " + generated_file)

    return



if __name__ == "__main__":
    args = parse_args()
    main(args)
