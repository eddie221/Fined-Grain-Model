#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:04:54 2020

@author: eddie
"""
from .resnet import resnet50
import torch.nn as nn
import torch

class Model_Net(nn.Module):
    def __init__(self, num_classes):
        self.backbone1 = resnet50(num_classes)
        self.backbone2 = resnet50(num_classes)
        
    def forward(self, x):
        if x.get_device() == -1:
            device = 'cpu'
        else:
            device = x.get_device()
        _, cam, cam_rf = self.backbone1(x)
        mask = torch.where(cam > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))
        
        mask_x = mask * x
        
        result, _, _ = self.backbone2(x)
        return result, cam, cam_rf
