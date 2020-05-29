# Inspired by - https://github.com/kuangliu/pytorch-cifar

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Hashed import HashedConv2d

class ShortCut(nn.Module):
    def __init__(self, in_planes, expansion_planes, kernel_size=1, stride=1, bias=False, buckets=16):
        super(ShortCut, self).__init__()
        self.conv = HashedConv2d(in_planes, expansion_planes, kernel_size=kernel_size, stride=stride, 
                                 bias=bias, buckets=buckets)
        self.bn = nn.BatchNorm2d(expansion_planes)
        
    def forward(self, x, x_hash):
        out, out_hash = self.conv(x, x_hash)
        out = self.bn(out)
        out_hash = self.bn(out_hash)
        return out, out_hash
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x, x_hash):
        return x, x_hash

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = HashedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                                  buckets=16)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = HashedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                                  buckets=16)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = Identity()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = ShortCut(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False,
                                     buckets=16)

    def forward(self, inp):
        x = inp[0]
        x_hash = inp[1]
        
        out, out_hash = self.conv1(x, x_hash)
        out = F.relu(self.bn1(out))
        out_hash = F.relu(self.bn1(out_hash))
        
        out, out_hash = self.conv2(out, out_hash)
        out = self.bn2(out)
        out_hash = self.bn2(out_hash)
        
        x, x_hash = self.shortcut(x, x_hash)
        out += x
        out_hash += x_hash
        out = F.relu(out)
        out_hash = F.relu(out_hash)
        return {0:out, 1:out_hash}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = HashedConv2d(in_planes, planes, kernel_size=1, bias=False,
                                  buckets=16)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = HashedConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                                  buckets=16)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = HashedConv2d(planes, self.expansion * planes, kernel_size=1, bias=False,
                                  buckets=16)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = Identity()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = ShortCut(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False,
                                     buckets=16)

    def forward(self, inp):
        x = inp[0]
        x_hash = inp[1]
        
        out, out_hash = self.conv1(x, x_hash)
        out = F.relu(self.bn1(out))
        out_hash = F.relu(self.bn1(out_hash))
        
        out, out_hash = self.conv2(out, out_hash)
        out = F.relu(self.bn2(out))
        out_hash = F.relu(self.bn2(out_hash))
        
        out, out_hash = self.conv3(out, out_hash)
        out = self.bn3(out)
        out_hash = self.bn3(out_hash)
        
        x, x_hash = self.shortcut(x, x_hash)
        out += x
        out_hash += x_hash
        out = F.relu(out)
        out_hash = F.relu(out_hash)
        return {0:out, 1:out_hash}


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = HashedConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False,
                                  buckets=16)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out, out_hash = self.conv1(x, x)
        out = F.relu(self.bn1(out))
        out_hash = F.relu(self.bn1(out_hash))

        tmp = {0:out, 1:out_hash}
        tmp = self.layer1(tmp)
        tmp = self.layer2(tmp)
        tmp = self.layer3(tmp)
        tmp = self.layer4(tmp)
        out = tmp[0]
        out_hash = tmp[1]
        
        out = F.avg_pool2d(out, 4)
        out_hash = F.avg_pool2d(out_hash, 4)
        out = out.view(out.size(0), -1)
        out_hash = out_hash.view(out_hash.size(0), -1)
        out = self.fc(out)
        out_hash = self.fc(out_hash)
        return out, out_hash


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])