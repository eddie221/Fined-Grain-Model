#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:34:25 2021

@author: mmplab603
"""

import torch
import torch.nn as nn
import neural_network.lifting_pool as lift_pool

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

class feature_block(nn.Module):
    def __init__(self, inplanes, planes, lifting = False):
        super(feature_block, self).__init__()
        self.lifting = lifting
        
        self.conv1 = nn.Conv2d(inplanes, planes, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(planes)
        
        if lifting:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes * 4, 1, stride = 2),
                                            nn.BatchNorm2d(planes * 4))
            self.avg = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(nn.Linear(planes * 4, planes * 4 // 8),
                                    nn.ReLU(),
                                    nn.Linear(planes * 4 // 8, planes * 4),
                                    nn.Sigmoid()
                                    )
            
            self.conv3 = nn.Conv2d(planes * 4, planes * 4, 1)
        else:
            self.conv3 = nn.Conv2d(planes, planes * 4, 1)
            
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU(inplace=True)  
        
    def attention(self, x):
        x_avg = self.avg(x)
        x_att = self.fc(x_avg.view(x.shape[0], -1))
        x_att = x_att.unsqueeze(2).unsqueeze(2)
        x = x_att * x + x
        return x
        
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        if self.lifting:
            x = lift_pool.lifting_down(x)
            x = torch.cat(x, dim = 1)
            x = self.attention(x)
            residual = self.downsample(residual)
            
        x = self.conv3(x)
        x = self.bn3(x)
        
        x = x + residual
        x = self.relu(x)
        return x
    
class dev_model(nn.Module):
    def __init__(self, num_classes):
        super(dev_model, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(self.inplanes, 64, 3)
        self.layer2 = self._make_layer(self.inplanes, 128, 4)
        self.layer3 = self._make_layer(self.inplanes, 256, 6)
        self.layer4 = self._make_layer(self.inplanes, 512, 3)
        
        self.FPN_feature = 512
        self.top = nn.Conv2d(2048, self.FPN_feature, kernel_size=1)
        self.lateral1 = lateral_connect(1024, self.FPN_feature)
        self.lateral2 = lateral_connect( 512, self.FPN_feature)
        
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc2 = self._construct_fc_layer([num_classes], 512)
        self.fc3 = self._construct_fc_layer([num_classes], 512)
        self.fc4 = self._construct_fc_layer([num_classes], 512)
        
    def _make_layer(self, inplanes, planes, blocks):
        layers = []
        layers.append(feature_block(inplanes, planes, True))
        inplanes = planes * 4
        for i in range(1, blocks):
            layers.append(feature_block(inplanes, planes))
        self.inplanes = inplanes
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
        
        x = self.layer1(x)
        x = self.layer2(x)
        x2 = x
        x = self.layer3(x)
        x3 = x
        x = self.layer4(x)
        x4 = x
        
        p2, p3, p4 = self.FPN(x2, x3, x4)
        p2 = self.avg(p2).view(p2.shape[0], -1)
        p3 = self.avg(p3).view(p3.shape[0], -1)
        p4 = self.avg(p4).view(p4.shape[0], -1)
        
        p2 = self.fc2(p2)
        p3 = self.fc3(p3)
        p4 = self.fc4(p4)
        return p2, p3, p4
    
    
def dev_mod(num_classes = 1000):
    model = dev_model(num_classes)
    return model

if __name__ == "__main__":
    model = dev_model(9).cuda()
    print(model)
    a = torch.randn([2, 3, 224, 224]).cuda()
    b = model(a)
    print(b)