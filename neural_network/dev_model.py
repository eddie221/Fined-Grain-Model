#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:34:25 2021

@author: mmplab603
"""

import torch
import torch.nn as nn
import neural_network.lifting_pool as lift_pool

class feature_block(nn.Module):
    def __init__(self, in_cha, out_cha):
        super(feature_block, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_cha, in_cha, 3, padding = 1),
                                   nn.BatchNorm2d(in_cha),
                                   nn.ReLU(),
                                   nn.Conv2d(in_cha, out_cha, 3, padding = 1),
                                   nn.BatchNorm2d(out_cha),
                                   nn.ReLU(),
                                   nn.Conv2d(out_cha, out_cha, 3, padding = 1),
                                   nn.BatchNorm2d(out_cha),
                                   )
        self.relu = nn.ReLU(inplace=True)    
        self.cha_equ = None
        if in_cha != out_cha:
            self.cha_equ = nn.Conv2d(in_cha, out_cha, 1)
        
    def forward(self, x):
        residual = x
        x = self.extract(residual)
        if self.cha_equ is not None:
            residual = self.cha_equ(residual)
            
        x = x + residual
        x = self.relu(x)
        return x
    
class lifting_extract(nn.Module):
    def __init__(self, in_cha, out_cha):
        super(lifting_extract, self).__init__()
        self.feature_extract_ll = feature_block(in_cha, out_cha)
        self.feature_extract_lh = feature_block(in_cha, out_cha)
        self.feature_extract_hl = feature_block(in_cha, out_cha)
        self.feature_extract_hh = feature_block(in_cha, out_cha)
        
    def forward(self, x):
        ll, lh, hl, hh = lift_pool.lifting_down(x)
        ll_x = self.feature_extract_ll(ll)
        lh_x = self.feature_extract_lh(lh)
        hl_x = self.feature_extract_hl(hl)
        hh_x = self.feature_extract_hh(hh)
        
        return ll_x, lh_x, hl_x, hh_x

class dev_model(nn.Module):
    def __init__(self, num_classes):
        super(dev_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.lifting_node_l1 = lifting_extract(64, 64)
        self.lifting_node_l2 = lifting_extract(256, 256)
        self.lifting_node_l3 = lifting_extract(1024, 1024)
        
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer([num_classes], 4096)
        
    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """
        Construct fully connected layer

        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        
        assert isinstance(fc_dims, (list, tuple)), "fc_dims must be either list or tuple, but got {}".format(type(fc_dims))
        
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        
        self.feature_dim = fc_dims[-1]
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layer1 = self.lifting_node_l1(x)
        layer1 = torch.cat(layer1, dim = 1)
        layer2 = self.lifting_node_l2(layer1)
        layer2 = torch.cat(layer2, dim = 1)
        layer3 = self.lifting_node_l3(layer2)
        layer3 = torch.cat(layer3, dim = 1)
    
        layer3 = self.avg(layer3)
        layer3 = layer3.view(layer3.shape[0], -1)
        layer3 = self.fc(layer3)
        return layer3
    
    
def dev_mod(num_classes = 1000):
    model = dev_model(num_classes)
    return model

if __name__ == "__main__":
    model = dev_model(9)
    a = torch.randn([2, 3, 224, 224])
    model(a)