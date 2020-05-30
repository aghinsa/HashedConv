# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class HashedConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.buckets = kwargs["buckets"]
        kwargs.pop("buckets")
        self.hash_value = nn.Parameter(torch.randn(self.buckets).sort().values)
        self.conv = nn.Conv2d(*args, **kwargs)
        self.conv_weight = nn.Parameter(torch.randn(self.conv.weight.size()))

    def forward(self, x, x_hash):
        max_weight = torch.max(self.conv_weight)
        min_weight = torch.min(self.conv_weight)
        size = self.conv.weight.size()
        self.hash_weight = self.conv_weight.clone()
        for m in range(size[0]):
            for c in range(size[1]):
                for r in range(size[2]):
                    for s in range(size[3]):
                        tmp = self.hash_weight[m][c][r][s]
                        for i in range(self.buckets):
                            th_high = (i + 1) * (
                                (max_weight - min_weight) / self.buckets
                            ) + min_weight
                            th_low = (i) * (
                                (max_weight - min_weight) / self.buckets
                            ) + min_weight
                            if th_high > tmp >= th_low:
                                self.hash_weight[m][c][r][s] = self.hash_value[i]
                            if tmp == max_weight:
                                self.hash_weight[m][c][r][s] = self.hash_value[
                                    self.buckets - 1
                                ]
        out = self.conv._conv_forward(x, self.conv_weight)
        hash_out = self.conv._conv_forward(x, self.hash_weight)
        out_hash = self.conv._conv_forward(x_hash, self.hash_weight)
        return out, hash_out, out_hash
