#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

import torchvision
import numpy as np


def show_batch(sample_batched, predictions=None, p_file='./test.png'):

    fig = plt.figure()
    if predictions is None:
        n_subplots = 2
    else:
        n_subplots = 3

    # Grid of images
    fig.add_subplot(n_subplots, 1, 1)
    create_display_grid(sample_batched['image'], normalize=True)

    # Grid of masks
    fig.add_subplot(n_subplots, 1, 2)
    create_display_grid(sample_batched['mask'][:, np.newaxis, ...])

    # Grid of predictions if predictions exist
    if predictions is not None:
        fig.add_subplot(n_subplots, 1, 3)
        create_display_grid(predictions.float())

    plt.savefig(p_file)
    plt.close()


def create_display_grid(data, normalize=False):
    grid = torchvision.utils.make_grid(data, normalize=normalize)
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
