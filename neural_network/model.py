#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:04:54 2020

@author: eddie
"""
from .classify import resnet50 as resnet50_classify
from .mask import resnet50 as resnet50_mask
import torch.nn as nn
import torch

class Model_Net(nn.Module):
    def __init__(self, num_classes):
        super(Model_Net, self).__init__()
        self.backbone1 = resnet50_mask(num_classes = num_classes)
        self.backbone2 = resnet50_classify(num_classes = num_classes)
        self.instance_norm = nn.InstanceNorm2d(1)
        self.relu = nn.ReLU()
    
    def feature_refined(self, cam):
        n, c, h, w = cam.shape
        cam = cam.view(n, -1, h * w)
        cam = cam / (torch.norm(cam, dim=1, keepdim = True) + 1e-5)
        
        correlation = self.relu(torch.matmul(cam.transpose(1, 2), cam))
        correlation = correlation / (torch.sum(correlation, dim = 1, keepdim = True) + 1e-5)
        cam_refined = torch.matmul(cam, correlation).view(n, -1, h, w)
        return cam_refined
# =============================================================================
#         B, C, H, W = cam.shape
#         cam_flatten = cam.contiguous().view(cam.shape[0], cam.shape[1], -1)
#         cam_flatten = cam_flatten / torch.norm(cam_flatten, dim = 2, keepdim = True)
#         cam_cor = torch.matmul(cam_flatten, cam_flatten.transpose(1, 2))
#         cam_cor = self.relu(cam_cor)
#         cam_cor = cam_cor / (torch.sum(cam_cor, dim = 1, keepdim = True) + 1e-5)
#         cam_flatten = cam.contiguous().view(cam.shape[0], cam.shape[1], -1)
#         
#         cam_refined = torch.matmul(cam_flatten.transpose(1, 2), cam_cor).transpose(1, 2).contiguous().view(B, C, H, W)
#         return cam_refined
# =============================================================================
        
    def forward(self, x):
        if x.get_device() == -1:
            device = 'cpu'
        else:
            device = x.get_device()
        
        # mask model ----------------------------------------------------------
        result_1, feature_1 = self.backbone1(x)

        # get weight
        predict_class = torch.max(result_1, dim = 1)[1]
        channel_weight = self.backbone1.state_dict()['fc.weight']
        with torch.no_grad():
            # create cam
            cam_1 = feature_1 * channel_weight[predict_class].unsqueeze(2).unsqueeze(3)
            cam_1 = nn.functional.interpolate(torch.sum(cam_1, dim = 1, keepdim = True), size = x.shape[2], mode = 'bilinear', align_corners = True)
            cam_1 = self.instance_norm(cam_1)
            
            # refine feature
            feature_rf_1 = self.feature_refined(feature_1)
            cam_rf_1 = feature_rf_1 * channel_weight[predict_class].unsqueeze(2).unsqueeze(3)
            cam_rf_1 = nn.functional.interpolate(torch.sum(cam_rf_1, dim = 1, keepdim = True), size = x.shape[2], mode = 'bilinear', align_corners = True)
            cam_rf_1 = self.instance_norm(cam_rf_1)
            # ---------------------------------------------------------------------
            mask = torch.where(cam_1 > 0.5, torch.tensor(1.).to(device), torch.tensor(0.).to(device))
            mask_x = x * mask
        
        # classify model ------------------------------------------------------
        result_2, feature_2 = self.backbone2(mask_x.detach())
        
        # get weight
        predict_class = torch.max(result_2, dim = 1)[1]
        channel_weight = self.backbone2.state_dict()['fc.weight']
        with torch.no_grad():
            # create cam
            cam_2 = feature_2 * channel_weight[predict_class].unsqueeze(2).unsqueeze(3)
            cam_2 = nn.functional.interpolate(torch.sum(cam_2, dim = 1, keepdim = True), size = x.shape[2], mode = 'bilinear', align_corners = True)
            cam_2 = self.instance_norm(cam_2)
            
            # refine feature
            feature_rf_2 = self.feature_refined(feature_2)
            cam_rf_2 = feature_rf_2 * channel_weight[predict_class].unsqueeze(2).unsqueeze(3)
            cam_rf_2 = nn.functional.interpolate(torch.sum(cam_rf_2, dim = 1, keepdim = True), size = x.shape[2], mode = 'bilinear', align_corners = True)
            cam_rf_2 = self.instance_norm(cam_rf_2)
        
        return result_1, result_2, cam_1, cam_rf_1, cam_2, cam_rf_2
