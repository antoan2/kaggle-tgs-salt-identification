import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from skimage.color import gray2rgb
import numpy as np
import os
from skimage import io
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
from utils.submission import rle_encode, rle_decode, write_results_file


def crf(original_image, mask_img):

    # Converting annotated image to RGB if it is Gray scale
    if (len(mask_img.shape) < 3):
        mask_img = gray2rgb(mask_img)


#     #Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask_img[:, :, 0] + (mask_img[:, :, 1] << 8) + (
        mask_img[:, :, 2] << 16)

    #     # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2

    #Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0],
                        n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(
        sxy=(3, 3),
        compat=3,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC)

    #Run Inference for 10 steps
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0], original_image.shape[1]))


def read_submission_file(path):
    results = {}
    with open(path) as file_id:
        file_id.readline()
        for line in file_id:
            image_name, rle_code = line.strip().split(',')
            results[image_name] = rle_code
    return results


if __name__ == "__main__":
    results = read_submission_file('./predictions.csv')
    results_post_processed = {}
    for image_name, rle_code in tqdm(results.items()):
        mask = rle_decode(rle_code)
        img = io.imread(os.path.join('/data/test/images', image_name + '.png'))
        mask_post_processed = crf(img, mask)
        """
        fig = plt.figure()
        fig.add_subplot(121)
        plt.imshow(mask)
        fig.add_subplot(122)
        plt.imshow(mask_post_processed)
        plt.savefig('./postprocessing.png')
        """
        results_post_processed[image_name] = rle_encode(mask_post_processed)
    write_results_file(results_post_processed, './predictions_post_processing.csv')
