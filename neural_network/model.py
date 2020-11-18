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
        super(Model_Net, self).__init__()
        self.backbone1 = resnet50(num_classes = num_classes)
        self.backbone2 = resnet50(num_classes = num_classes)
        
    def forward(self, x):
        if x.get_device() == -1:
            device = 'cpu'
        else:
            device = x.get_device()
        result_1, cam_1, cam_rf_1 = self.backbone1(x)
        mask = torch.where(cam_1 > 0.5, torch.tensor(1.).to(device), torch.tensor(0.).to(device))
        
        mask = torch.nn.functional.interpolate(mask, size = x.shape[2], mode = 'bilinear', align_corners = True)
        
        mask_x = mask * x
        
        result_2, cam_2, cam_rf_2 = self.backbone2(mask_x.detach())
        return result_1, result_2, cam_1, cam_rf_1, cam_2, cam_rf_2
