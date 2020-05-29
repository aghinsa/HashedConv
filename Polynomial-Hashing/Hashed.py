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
        self.values = torch.zeros(self.buckets, 1).cuda()
        b = int(self.buckets/2)
        for i in range(self.buckets):
            if(i < b):
                self.values[i,0] = 1/-2**(b-i)
            else:
                self.values[i,0] = 1/2**(i-b)
        self.a0 = nn.Parameter(torch.randn(self.buckets,1))
        self.a1 = nn.Parameter(torch.randn(self.buckets,1))
        self.a2 = nn.Parameter(torch.randn(self.buckets,1))
        self.conv = nn.Conv2d(*args, **kwargs)
        self.weight = nn.Parameter(self.conv.weight)
        self.softmax = nn.Softmax(dim=0)
        self.T = 100
        
    def forward(self, x, x_hash):
        N, C, H, W = self.weight.shape
        cw = self.weight.clone().reshape(1,N*C*H*W)
        self.fx = torch.add(torch.mm(self.a1, cw) + torch.mm(self.a2, torch.pow(cw,2)), self.a0)
        self.hash_weight = torch.mm(self.softmax(self.fx*self.T).t(), self.values).reshape(N,C,H,W)
        out = self.conv._conv_forward(x, self.weight)
        out_hash = self.conv._conv_forward(x_hash, self.hash_weight)
        return out, out_hash

class HashedLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.buckets = kwargs["buckets"]
        kwargs.pop("buckets")
        self.values = torch.zeros(self.buckets, 1).cuda()
        b = int(self.buckets/2)
        for i in range(self.buckets):
            if(i < b):
                self.values[i,0] = 1/2**(b-i)
            else:
                self.values[i,0] = 1/-2**(i-b)
        self.a0 = nn.Parameter(torch.randn(self.buckets,1))
        self.a1 = nn.Parameter(torch.randn(self.buckets,1))
        self.a2 = nn.Parameter(torch.randn(self.buckets,1))
        self.fc = nn.Linear(*args, **kwargs)
        self.weight = nn.Parameter(self.fc.weight)
        self.bias = nn.Parameter(self.fc.bias)
        self.softmax = nn.Softmax(dim=0)
        self.T = 100
        
    def forward(self, x, x_hash):
        N, M = self.weight.shape
        fcw = self.weight.clone().reshape(1,N*M)
        self.fx = torch.add(torch.mm(self.a1, fcw) + torch.mm(self.a2, torch.pow(fcw,2)), self.a0)
        self.hash_weight = torch.mm(self.softmax(self.fx*self.T).t(), self.values).reshape(N,M)
        out = torch.add(torch.matmul(x, self.weight.t()), self.bias)
        out_hash = torch.add(torch.matmul(x_hash, self.hash_weight.t()), self.bias)
        return out, out_hash