#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models
#import sys
#sys.path.append("/home/ziniuw/Fixed_Kernel_CNN/models_8K")
#from sync_batchnorm import SynchronizedBatchNorm2d



class FKCNN_2l(nn.Module):
    def __init__(self, model_arch, input_channel=1, input_size=64, output_size=1):
        super(FKCNN, self).__init__()
        l1_c = model_arch['l1_c']
        l1_k = model_arch['l1_k']
        p1_k = model_arch['p1_k']
        p1_s = model_arch['p1_s']
        l2_c = model_arch['l2_c']
        l2_k = model_arch['l2_k']
        l2_s = model_arch['l2_s']
        p2_k = model_arch['p2_k']
        p2_s = model_arch['p2_s']
        self.first_layer = nn.Conv2d(input_channel, l1_c, l1_k, bias=False)
        self.conv = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm2d(l1_c)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(p1_k, stride=p1_s)),
            ('conv2', nn.Conv2d(l1_c, l2_c, l2_k, stride=l2_s, bias=False)),
            ('bn2', SynchronizedBatchNorm2d(l2_c)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(p2_k, stride=p2_s))
        ]))
        flatten_size = (((input_size - l1_k + 1 - p1_k) // p1_s + 1 - l2_k)//l2_s + 1 - p2_k) // p2_s + 1
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(16 * flatten_size * flatten_size, 300)),
            ('relu', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(300, output_size))
        ]))

    def forward(self, x):
        x = self.first_layer(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class FKCNN_3l(nn.Module):
    def __init__(self, model_arch, input_channel=1, input_size=128, output_size=1):
        super(FKCNN_3l, self).__init__()
        l1_c = model_arch['l1_c']
        l1_k = model_arch['l1_k']
        p1_k = model_arch['p1_k']
        p1_s = model_arch['p1_s']
        l2_c = model_arch['l2_c']
        l2_k = model_arch['l2_k']
        l2_s = model_arch['l2_s']
        p2_k = model_arch['p2_k']
        p2_s = model_arch['p2_s']
        l3_c = model_arch['l3_c']
        l3_k = model_arch['l3_k']
        l3_s = model_arch['l3_s']
        p3_k = model_arch['p3_k']
        p3_s = model_arch['p3_s']
        self.first_layer = nn.Conv2d(input_channel, l1_c, l1_k, bias=False)
        self.conv = nn.Sequential(OrderedDict([
            ('bn1', nn.BatchNorm2d(l1_c)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(p1_k, stride=p1_s)),
            ('conv2', nn.Conv2d(l1_c, l2_c, l2_k, stride=l2_s, bias=False)),
            ('bn2', nn.BatchNorm2d(l2_c)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(p2_k, stride=p2_s)),
            ('conv3', nn.Conv2d(l2_c, l3_c, l3_k, stride=l3_s, bias=False)),
            ('bn3', nn.BatchNorm2d(l3_c)),
            ('relu3', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool2d(p3_k, stride=p3_s))
        ]))
        flatten_size = (((((input_size - l1_k + 1 - p1_k) // p1_s + 1 - l2_k)//l2_s + 1 - p2_k) // p2_s + 1 - 
                         l3_k) // l3_s + 1 - p3_k) // p3_s + 1
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(l3_c * flatten_size * flatten_size, 300)),
            ('relu', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(300, output_size))
        ]))

    def forward(self, x):
        x = self.first_layer(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
