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
    def __init__(self, num_classes, top_n = 3):
        super(Model_Net, self).__init__()
        self.backbone1 = resnet50_mask(num_classes = num_classes)
        self.backbone2 = resnet50_classify(num_classes = num_classes)
        self.instance_norm_2 = nn.InstanceNorm2d(1)
        self.instance_norm_3 = nn.InstanceNorm2d(1)
        self.instance_norm_4 = nn.InstanceNorm2d(1)
        self.relu = nn.ReLU()
        self.top_n = top_n
    
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

    def create_cam(self, feature, channel_weight, size):
        feature_rf = self.feature_refined(feature)
        
        # create cam
        cam = feature * channel_weight[:, 0].unsqueeze(2).unsqueeze(3)
        # refine feature
        cam_rf = feature_rf * channel_weight[:, 0].unsqueeze(2).unsqueeze(3)
        
        for i in range(1, self.top_n):
            with torch.no_grad():
                # create cam
                tmp_cam = feature * channel_weight[:, i].unsqueeze(2).unsqueeze(3)
                
                # refine feature
                tmp_cam_rf = feature_rf * channel_weight[:, i].unsqueeze(2).unsqueeze(3)
            
            if feature.size(2) == size // 8:
                cam = self.instance_norm_2(tmp_cam) + cam
                cam_rf = self.instance_norm_2(tmp_cam_rf) + cam_rf
                
            elif feature.size(2) == size // 16:
                cam = self.instance_norm_3(tmp_cam) + cam
                cam_rf = self.instance_norm_3(tmp_cam_rf) + cam_rf
                
            elif feature.size(2) == size // 32:
                cam = self.instance_norm_4(tmp_cam) + cam
                cam_rf = self.instance_norm_4(tmp_cam_rf) + cam_rf

        cam = nn.functional.interpolate(torch.sum(cam, dim = 1, keepdim = True), size = size, mode = 'bilinear', align_corners = True)
        cam_rf = nn.functional.interpolate(torch.sum(cam_rf, dim = 1, keepdim = True), size = size, mode = 'bilinear', align_corners = True)
            
        return cam, cam_rf

    def forward(self, x):
        if x.get_device() == -1:
            device = 'cpu'
        else:
            device = x.get_device()
        
        # mask model ----------------------------------------------------------
        result_1, x4_cls, x34_cls, x234_cls, x4, x34, x234 = self.backbone1(x)

        # get weight
        _, class_sort = torch.sort(result_1, dim = 1, descending = True)
        class_sort = class_sort[:,:self.top_n]
        with torch.no_grad():
            cam_1_4, cam_rf_1_4 = self.create_cam(x4, self.backbone1.state_dict()['fc_4.weight'][class_sort], x.shape[2])
            cam_1_34, cam_rf_1_34 = self.create_cam(x34, self.backbone1.state_dict()['fc_34.weight'][class_sort], x.shape[2])
            cam_1_234, cam_rf_1_234 = self.create_cam(x234, self.backbone1.state_dict()['fc_234.weight'][class_sort], x.shape[2])
            
            cam_1 = (cam_1_4 + cam_1_34 + cam_1_234) / 3
            cam_rf_1 = (cam_rf_1_4 + cam_rf_1_34 + cam_rf_1_234) / 3
            
            mask = torch.where(cam_1 > 0.5, torch.tensor(1.).to(device), torch.tensor(0.).to(device))
            mask_x = x * mask
        
        # classify model ------------------------------------------------------
        result_2 = self.backbone2(mask_x.detach())
        
        
        return [result_1, x4_cls, x34_cls, x234_cls], result_2, cam_1, cam_rf_1
