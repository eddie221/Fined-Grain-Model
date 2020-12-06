#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 15:57:46 2020

@author: eddie
"""

import torch.nn as nn
import torch

class Graph_nn(nn.Module):
    def __init__(self, cha, layer = 1):
        super(Graph_nn, self).__init__()
        self.param = nn.Parameter(torch.randn([layer, cha, cha, 1])).cuda()
        self.layer = layer
        self.instance_norm = nn.InstanceNorm1d(cha)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        if len(x.shape) == 3:
            batch, channel, feature_dim = x.shape   
        elif len(x.shape) == 4:
             batch, channel, height, width = x.shape   
             
        x = x.view(batch, channel, 1, -1)
        
        for i in range(self.layer):
            x = x * self.param[i : i + 1, :]
            x = torch.sum(x, dim = 1)
            x = self.instance_norm(x)
            x = self.relu(x)
            x = x.view(batch, channel, 1, -1)
        
        #x = torch.sum(x, dim = 1)
        return x