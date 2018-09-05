#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Naive(nn.Module):
    def __init__(self):
        super(Naive, self).__init__()
        # self.maxpool = nn.MaxPool2d(2, padding=1)
        self.activation = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=11, padding=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)

        x = self.output(x)
        return x
