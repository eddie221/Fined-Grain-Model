#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:34:25 2021

@author: mmplab603
"""

import torch
import torch.nn as nn
import neural_network.lifting_pool as lift_pool

class lifting_extract(nn.Module):
    def __init__(self, in_cha, out_cha, **kwargs):
        super(lifting_extract, self).__init__()
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
        
        return x
    
class lateral_connect(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(lateral_connect, self).__init__()
        self.lateral = nn.Conv2d(input_channel, output_channel, kernel_size=1)
        self.append_conv = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
    
    def upsample_and_add(self, target, small):
        n, c, h, w = target.size()
        return torch.nn.functional.interpolate(small, size=(h, w), mode='bilinear', align_corners=True) + target
    
    def forward(self, bottom_up, top_down):
        return self.append_conv(self.upsample_and_add(self.lateral(bottom_up), top_down))

class dev_model(nn.Module):
    def __init__(self, num_classes):
        super(dev_model, self).__init__()
        self.in_cha = 16
        self.conv1 = nn.Conv2d(3, self.in_cha, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_cha)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lifting_node_l1 = self._make_layer(lifting_extract, self.in_cha, self.in_cha, 3)
        self.lifting_node_l2 = self._make_layer(lifting_extract, self.in_cha, self.in_cha, 4)
        self.lifting_node_l3 = self._make_layer(lifting_extract, self.in_cha, self.in_cha, 6)
        self.lifting_node_l4 = self._make_layer(lifting_extract, self.in_cha, self.in_cha, 3)
        
        self.FPN_feature = 256
        self.top = nn.Conv2d(4096, self.FPN_feature, kernel_size=1)
        self.lateral1 = lateral_connect(1024, self.FPN_feature)
        self.lateral2 = lateral_connect( 256, self.FPN_feature)
        
        self.avg = nn.AdaptiveAvgPool2d(1)
        
        self.fc2 = self._construct_fc_layer([num_classes], 256)
        self.fc3 = self._construct_fc_layer([num_classes], 256)
        self.fc4 = self._construct_fc_layer([num_classes], 256)
        
    def _make_layer(self, block, in_cha, out_cha, layer, **kwargs):
        layers = []
        for i in range(layer - 1):
            layers.append(block(in_cha, in_cha, **kwargs))
        
        layers.append(block(in_cha, out_cha, **kwargs))
        self.in_cha = self.in_cha * 4
        return nn.Sequential(*layers)
        
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
    
    def FPN(self, x2, x3, x4):
        p4 = self.top(x4)
        p3 = self.lateral1(x3, p4)
        p2 = self.lateral2(x2, p3)
        return p2, p3, p4
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.lifting_node_l1(x)
        layer1 = lift_pool.lifting_down(layer1)
        layer1 = torch.cat(layer1, dim = 1)
        
        layer2 = self.lifting_node_l2(layer1)
        layer2 = lift_pool.lifting_down(layer2)
        layer2 = torch.cat(layer2, dim = 1)
        
        layer3 = self.lifting_node_l3(layer2)
        layer3 = lift_pool.lifting_down(layer3)
        layer3 = torch.cat(layer3, dim = 1)

        layer4 = self.lifting_node_l4(layer3)
        layer4 = lift_pool.lifting_down(layer4)
        layer4 = torch.cat(layer4, dim = 1)
        
        f2, f3, f4 = self.FPN(layer2, layer3, layer4)
        
        f2 = self.avg(f2)
        f2 = f2.view(f2.shape[0], -1)
        f2 = self.fc2(f2)
        
        f3 = self.avg(f3)
        f3 = f3.view(f3.shape[0], -1)
        f3 = self.fc3(f3)
        
        f4 = self.avg(f4)
        f4 = f4.view(f4.shape[0], -1)
        f4 = self.fc4(f4)
        return f2, f3, f4
    
    
def dev_mod(num_classes = 1000):
    model = dev_model(num_classes)
    return model

if __name__ == "__main__":
    model = dev_model(9)
    a = torch.randn([2, 3, 224, 224])
    model(a)